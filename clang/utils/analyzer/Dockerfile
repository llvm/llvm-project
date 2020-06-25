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
    git \
    gettext \
    python3 \
    python3-pip \
    cmake \
    ninja-build

# box2d dependencies
RUN apt-get install -y \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev

# symengine dependencies
RUN apt-get install -y \
    libgmp10 \
    libgmp-dev

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
