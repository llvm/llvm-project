FROM docker.io/python:3.12@sha256:ed942629d18ad03521f9835ff95f3edbfbe99ccd38be6ba64a509ce3c1b149a8

COPY requirements.lock.txt ./
RUN pip3 install --no-cache-dir -r requirements.lock.txt
COPY metrics.py ./

CMD ["python3", "metrics.py"]
