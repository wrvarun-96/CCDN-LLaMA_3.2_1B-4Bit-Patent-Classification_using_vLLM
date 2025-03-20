import os
import streamlit as st
import numpy as np
from vllm import LLM
from huggingface_hub import snapshot_download, login

# ✅ Set Streamlit Page Config (FIRST COMMAND)
st.set_page_config(page_title="LLM Classifier", layout="centered")

# ✅ Initialize session state variables
if "hf_token" not in st.session_state:
    st.session_state["hf_token"] = None
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "llm" not in st.session_state:
    st.session_state["llm"] = None  # Model will be stored here after loading

# ✅ Hugging Face Login Page
def login_page():
    """Hugging Face Token Login Page"""
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>🔑 Hugging Face Login</h2>", unsafe_allow_html=True)
    
    hf_token = st.text_input("Enter your Hugging Face Token:", type="password")

    if st.button("🔓 Login to Hugging Face"):
        if hf_token:
            login(token=hf_token)
            st.success("✅ Successfully logged into Hugging Face!")
            st.session_state["hf_token"] = hf_token  # ✅ Store token securely
            st.session_state["authenticated"] = True
            st.rerun()  # ✅ Refresh Streamlit UI to show the prediction page
        else:
            st.warning("⚠️ Please enter a valid token.")

# ✅ Prediction Page
def prediction_page():
    """LLM Text Classification Page (Only visible after login)"""
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>🧠 AI-Powered Text Classifier</h2>", unsafe_allow_html=True)
    st.write("Enter text below to classify it into one of the 9 categories.")

    # ✅ Retrieve token from session state
    hf_token = st.session_state["hf_token"]

    # ✅ Define Hugging Face Model Repo
    hf_model_repo = "Varu96/Llama3.2-1B-4Bit-Merged"
    local_model_dir = "hf_model"

    # ✅ Download model ONCE with authentication
    if not st.session_state["model_loaded"]:
        with st.spinner("🔄 Downloading merged model from Hugging Face..."):
            snapshot_download(repo_id=hf_model_repo, local_dir=local_model_dir, token=hf_token)
            st.success("✅ Model downloaded successfully!")
            st.session_state["model_loaded"] = True  # ✅ Model is now loaded

    # ✅ Load vLLM model only once
    if st.session_state["llm"] is None:
        with st.spinner("🚀 Loading Model... This may take a few seconds."):
            st.session_state["llm"] = LLM(
                model=local_model_dir,
                task="classify",
                enforce_eager=True
            )
            st.success("✅ Model is ready!")

    # ✅ Get LLM model from session state
    llm = st.session_state["llm"]

    # 🔹 User Input Box
    user_input = st.text_area("📝 Enter your text here:", height=150)

    if st.button("🔍 Classify Text"):
        if user_input.strip():
            # ✅ Run classification using vLLM
            output = llm.classify([user_input])[0]
            probs = np.array(output.outputs.probs)
            predicted_class = probs.argmax()
            st.success(f"🗂 **Predicted Category:** {predicted_class}")
        else:
            st.warning("⚠️ Please enter some text to classify!")

# ✅ Logic to Switch Between Login & Prediction Page
if not st.session_state["authenticated"]:
    login_page()  # ✅ Show Login Page First
else:
    prediction_page()  # ✅ Show Prediction Page After Login
