FROM docker.io/library/ubuntu:24.04

# MPFR is required by some of the mathlib tests.
RUN apt-get update && \
    apt-get install -y \
    git \
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
    cmake \
    linux-libc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    sudo ./llvm.sh 21 && \
    rm llvm.sh
    
# Debian has a multilib setup, so we need to symlink the asm directory.
# For more information, see https://wiki.debian.org/Multiarch/LibraryPathOverview
RUN ln -sf /usr/include/$(uname -p)-linux-gnu/asm /usr/include/asm

# Create a new user to avoid test failures related to a lack of expected
# permissions issues in some tests. Set the user id to 1001 as that is the
# user id that Github Actions uses to perform the checkout action.
RUN useradd gha -u 1001 -m -s /bin/bash

# Also add the user to passwordless sudoers so that we can install software
# later on without having to rebuild the container.
RUN adduser gha sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER gha
WORKDIR /home/gha
