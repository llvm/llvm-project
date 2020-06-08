FROM ubuntu:bionic

RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    wget

# newer CMake is required by LLVM
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ bionic main'

# test system dependencies
RUN apt-get update && apt-get install -y \
    git=1:2.17.1-1ubuntu0.7 \
    gettext=0.19.8.1-6ubuntu0.3 \
    python3=3.6.7-1~18.04 \
    python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
    cmake=3.17.3-0kitware1 \
    ninja-build=1.8.2-1

# box2d dependencies
RUN apt-get install -y \
    libx11-dev=2:1.6.4-3ubuntu0.2 \
    libxrandr-dev=2:1.5.1-1 \
    libxinerama-dev=2:1.1.3-1 \
    libxcursor-dev=1:1.1.15-1 \
    libxi-dev=2:1.7.9-1

# symengine dependencies
RUN apt-get install -y \
    libgmp10=2:6.1.2+dfsg-2 \
    libgmp-dev=2:6.1.2+dfsg-2

# simbody dependencies
RUN apt-get install -y \
    liblapack-dev=3.7.1-4ubuntu1

# drogon dependencies
RUN apt-get install -y \
    libjsonrpccpp-dev=0.7.0-1build2 \
    uuid-dev=2.31.1-0.4ubuntu3.6

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

VOLUME /analyzer
VOLUME /projects
VOLUME /llvm-project
VOLUME /build
VOLUME /scripts

ENV PATH="/analyzer/bin:${PATH}"

ADD entrypoint.py /entrypoint.py

# Uncomment in case of requirements
# ADD requirements.txt /requirements.txt
# RUN pip3 install -r /requirements.txt

ENTRYPOINT ["python", "/entrypoint.py"]
