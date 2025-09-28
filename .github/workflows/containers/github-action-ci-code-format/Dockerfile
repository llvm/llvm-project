FROM docker.io/library/ubuntu:24.04 AS base
ENV LLVM_SYSROOT=/opt/llvm

FROM base AS clang-format-toolchain
ENV LLVM_VERSION=21.1.1

RUN apt-get update && \
    apt-get install -y \
    wget \
    gcc \
    g++ \
    cmake \
    ninja-build \
    python3 \
    git \
    curl \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O -L https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-$LLVM_VERSION.tar.gz && \
  tar -xf llvmorg-$LLVM_VERSION.tar.gz && \
  rm -f llvmorg-$LLVM_VERSION.tar.gz

WORKDIR /llvm-project-llvmorg-$LLVM_VERSION

RUN cmake -B ./build -G Ninja ./llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_SYSROOT" \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DLLVM_DISTRIBUTION_COMPONENTS="clang-format"

RUN ninja -C ./build install-distribution

FROM base AS ci-container-code-format

COPY --from=clang-format-toolchain $LLVM_SYSROOT $LLVM_SYSROOT

# Need nodejs for some of the GitHub actions.
# Need git for git-clang-format.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # binutils \
    git \
    nodejs \
    # python3-psutil \
    sudo \
    # These are needed by the premerge pipeline. Pip and venv are used to
    # install dependent python packages.
    # Having a symlink from python to python3 enables code sharing between
    # the Linux and Windows pipelines.
    python3-pip \
    python3-venv \
    python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LLVM_SYSROOT=$LLVM_SYSROOT
ENV PATH=${LLVM_SYSROOT}/bin:${PATH}

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

COPY llvm/utils/git/requirements_formatting.txt /home/gha/requirements_formatting.txt
RUN python -m venv venv && \
    venv/bin/pip install -r /home/gha/requirements_formatting.txt && \
    rm /home/gha/requirements_formatting.txt
