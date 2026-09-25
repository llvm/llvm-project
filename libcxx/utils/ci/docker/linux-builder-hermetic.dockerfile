# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
#
# This file defines the hermetic Linux builder image used for libc++ CI testing
# where CMake, Ninja, and Clang/LLVM are placed under /opt/hermetic outside of
# PATH, catching any implicit dependencies on host build tools or compilers.
#
# From the root of the monorepo, this image can be built with:
#
#   $ docker compose --file libcxx/utils/ci/docker/docker-compose.yml build libcxx-linux-builder-hermetic
#

FROM docker.io/library/ubuntu:26.04

# Changing this file causes a rebuild of the image in a GitHub action.
RUN echo "Last forced update executed on 2026-09-24."

# Make sure apt-get doesn't try to prompt for stuff like our time zone, etc.
ENV DEBIAN_FRONTEND=noninteractive

# Populated in the docker-compose file
ARG LLVM_HEAD_VERSION=23
ENV LLVM_HEAD_VERSION=${LLVM_HEAD_VERSION}

ARG CMAKE_VERSION=4.4.3
ENV CMAKE_VERSION=${CMAKE_VERSION}

ARG NINJA_VERSION=1.13.2
ENV NINJA_VERSION=${NINJA_VERSION}

# Install sudo and setup passwordless sudo.
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "ALL ALL = (ALL) NOPASSWD: ALL" | tee /etc/sudoers

# Installing tzdata before other packages avoids the time zone prompts.
RUN sudo apt-get update \
    && sudo apt-get install -y \
        tzdata

# Install runtime/test utilities and the basic host C sysroot (libc6-dev, libgcc-15-dev).
# Intentionally OMIT: build-essential, gcc, g++, clang, make, cmake, ninja-build.
RUN sudo apt-get update \
    && sudo apt-get install -y \
        bash \
        binutils \
        bzip2 \
        curl \
        gdb \
        git \
        gpg \
        language-pack-en \
        language-pack-fr \
        language-pack-ja \
        language-pack-ru \
        language-pack-zh-hans \
        libc6-dev \
        libgcc-15-dev \
        libstdc++-15-dev \
        lsb-release \
        python3 \
        python3-dev \
        python3-packaging \
        python3-psutil \
        python3-setuptools \
        python3-venv \
        python3-yaml \
        rsync \
        unzip \
        wget \
        xz-utils \
    && sudo rm -rf /var/lib/apt/lists/*

# These two locales are not enabled by default so generate them
RUN printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2\n" | sudo tee -a /etc/locale.gen && \
    sudo mkdir -p /usr/local/share/i1en/ && \
    printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2\n" | sudo tee -a /usr/local/share/i1en/SUPPORTED && \
    sudo locale-gen

# Fetch CMake and Ninja from official GitHub releases, and Clang from Compiler Explorer
# (matching linux-builder.dockerfile's ce_install and LLVM_HEAD_VERSION), into /opt/hermetic
# outside of PATH.
RUN sudo mkdir -p /opt/hermetic/cmake /opt/hermetic/ninja && \
    curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
      | sudo tar xzf - --strip-components=1 -C /opt/hermetic/cmake && \
    curl -fsSL -o /tmp/ninja-linux.zip \
      "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip" && \
    sudo unzip -q /tmp/ninja-linux.zip -d /opt/hermetic/ninja && \
    sudo chmod +x /opt/hermetic/ninja/ninja && \
    rm -f /tmp/ninja-linux.zip && \
    sudo apt-get update && sudo apt-get install -y make && \
    sudo git clone --depth 1 https://github.com/compiler-explorer/infra.git /tmp/ce-infra && \
    (cd /tmp/ce-infra && sudo make ce) && \
    sudo /tmp/ce-infra/bin/ce_install --dest /opt/hermetic install compilers/c++/clang $((LLVM_HEAD_VERSION - 1)).1.0 && \
    sudo ln -sfn /opt/hermetic/clang-$((LLVM_HEAD_VERSION - 1)).1.0 /opt/hermetic/llvm && \
    sudo rm -rf /tmp/ce-infra && \
    sudo apt-get purge -y make && sudo apt-get autoremove -y && sudo rm -rf /var/lib/apt/lists/*
