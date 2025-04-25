import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
reverse_word_index = {value : key for key,value in word_index.items()}

model = load_model('Simple_RNN_Model_IMDB.h5')

def preprocessing_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict(review):
    preprocessed_input = preprocessing_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment,prediction[0][0]

st.title("IMDB Moive Review Sentiement Analysis")
st.write("Enter  movie review to classify it as positive or negative !")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    setniment,prediction = predict(user_input)
    st.write(prediction)
    st.write(setniment)
else:
    st.write("Please enter a movie review")

