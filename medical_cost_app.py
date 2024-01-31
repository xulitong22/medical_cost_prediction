import streamlit as st
import pandas as pd 
import numpy as np
import pickle

pd.set_option('future.no_silent_downcasting', True)
st.title('Medical Cost Prediction App')
st.write("""
This app predicts the medical cost by using linear regression
""")

st.write('---')

# Loads the medical cost dataset
df = pd.read_csv('data/insurance.csv')

st.sidebar.header('User input Parameters')

age = st.sidebar.slider('age', df['age'].min(), df['age'].max(), 40)
sex = st.sidebar.selectbox('sex', ('female', 'male'))
bmi = st.sidebar.slider('bmi', df['bmi'].min(), df['bmi'].max(), 29.81)
children = st.sidebar.selectbox('children', (0, 1, 2, 3, 4, 5))
smoker = st.sidebar.selectbox('smoker', ('yes', 'no'))

data = {'age': age, 
        'sex': sex, 
        'bmi': bmi, 
        'children': children, 
        'smoker': smoker}

user_df = pd.DataFrame([data])

st.header('Specified Input parameters')
st.write(user_df)
st.write('---')

user_df.replace({'yes': 1, 'no': 0, 'female': 0, 'male':1}, inplace=True)
X = user_df.values.reshape(1, -1)

# Load the saved model
with open('model_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

feature_names = user_df.columns  
X_named = pd.DataFrame(X, columns=feature_names)
prediction = loaded_pipeline.predict(X_named)

st.header('Predicted medical cost')
st.write(prediction)