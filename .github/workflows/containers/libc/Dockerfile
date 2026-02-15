FROM docker.io/library/ubuntu:24.04

RUN apt-get update && \
    apt-get install -y \
    libmpfr-dev \
    libgmp-dev \
    libmpc-dev \
    ninja-build \
    sudo \
    sccache \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    linux-libc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 21 && \
    rm llvm.sh

RUN ln -sf /usr/include/$(uname -p)-linux-gnu/asm /usr/include/asm
