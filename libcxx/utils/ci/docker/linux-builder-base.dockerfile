# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
#
# This file defines the base image we use for Linux testing using Github Actions.
# From the root of the monorepo, this image can be built with:
#
#   $ docker build --file libcxx/utils/ci/docker/linux-builder-base.dockerfile \
#                  --build-arg GCC_HEAD_VERSION=<version> \
#                  --build-arg LLVM_HEAD_VERSION=<version> .
#
# This image also gets built on every push to `main` that modifies these Docker
# files, and can be found at ghcr.io/libcxx/libcxx-linux-builder-base .

FROM docker.io/library/ubuntu:noble

# Changing this file causes a rebuild of the image in a GitHub action. However, it does not cause
# the CI runners to switch to that image automatically, that must be done by updating the image used
# by the libc++ self-hosted runners in llvm-zorg. The date uses the ISO format YYYY-MM-DD.
RUN echo "Last forced update executed on 2025-11-11."

# Make sure apt-get doesn't try to prompt for stuff like our time zone, etc.
ENV DEBIAN_FRONTEND=noninteractive

# populated in the docker-compose file
ARG GCC_HEAD_VERSION
ENV GCC_HEAD_VERSION=${GCC_HEAD_VERSION}

# populated in the docker-compose file
ARG LLVM_HEAD_VERSION
ENV LLVM_HEAD_VERSION=${LLVM_HEAD_VERSION}

# Install sudo and setup passwordless sudo.
RUN <<EOF
  apt-get update || true
  apt-get install -y sudo || true
  echo "ALL ALL = (ALL) NOPASSWD: ALL" | tee /etc/sudoers || true
EOF

# Installing tzdata before other packages avoids the time zone prompts.
# These prompts seem to ignore DEBIAN_FRONTEND=noninteractive.
RUN sudo apt-get update \
    && sudo apt-get install -y \
        tzdata

# Install various tools used by the build or the test suite
# TODO add ninja-build once 1.11 is available in Ubuntu, also remove the manual
# installation below.
RUN sudo apt-get update \
    && sudo apt-get install -y \
        bash \
        bzip2 \
        ccache \
        curl \
        gdb \
        git \
        gpg \
        language-pack-en \
        language-pack-fr \
        language-pack-ja \
        language-pack-ru \
        language-pack-zh-hans \
        libedit-dev \
        libncurses5-dev \
        libpython3-dev \
        libxml2-dev \
        lsb-release \
        make \
        ninja-build \
        python3 \
        python3-dev \
        python3-packaging \
        python3-setuptools \
        python3-psutil \
        software-properties-common \
        swig \
        unzip \
        uuid-dev \
        wget \
        xz-utils \
    && sudo rm -rf /var/lib/apt/lists/*

# These two locales are not enabled by default so generate them
RUN <<EOF
  set -e
  printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2" | sudo tee -a /etc/locale.gen
  sudo mkdir /usr/local/share/i1en/
  printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2" | sudo tee -a /usr/local/share/i1en/SUPPORTED
  sudo locale-gen
EOF

# Install Clang <latest>, <latest-1> and ToT, which are the ones we support.
# We also install <latest-2> because we need to support the "latest-1" of the
# current LLVM release branch, which is effectively the <latest-2> of the
# tip-of-trunk LLVM. For example, after branching LLVM 14 but before branching
# LLVM 15, we still need to have Clang 12 in this Docker image because the LLVM
# 14 release branch CI uses it. The tip-of-trunk CI will never use Clang 12,
# though.
RUN <<EOF
  set -e
  sudo apt-get update
  wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  sudo /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 3)) all  # for CI transitions
  sudo /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 2)) all  # previous release
  sudo /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 1)) all  # latest release
  sudo /tmp/llvm.sh $LLVM_HEAD_VERSION          all  # current ToT
  sudo rm -rf /var/lib/apt/lists/*
EOF

# Install the most recent GCC, like clang install the previous version as a transition.
RUN <<EOF
  set -e
  sudo git clone https://github.com/compiler-explorer/infra.git /tmp/ce-infra
  (cd /tmp/ce-infra && sudo make ce)
  # Current ToT, we do not guarantee any support in our support matrix.
  sudo /tmp/ce-infra/bin/ce_install --enable nightly install compilers/c++/nightly/gcc trunk
  sudo ln -s /opt/compiler-explorer/gcc-snapshot/bin/gcc /usr/bin/gcc-$GCC_HEAD_VERSION
  sudo ln -s /opt/compiler-explorer/gcc-snapshot/bin/g++ /usr/bin/g++-$GCC_HEAD_VERSION
  # The latest release.
  sudo /tmp/ce-infra/bin/ce_install install compilers/c++/x86/gcc $((GCC_HEAD_VERSION - 1)).1.0
  sudo ln -s /opt/compiler-explorer/gcc-$((GCC_HEAD_VERSION - 1)).1.0/bin/gcc /usr/bin/gcc-$((GCC_HEAD_VERSION - 1))
  sudo ln -s /opt/compiler-explorer/gcc-$((GCC_HEAD_VERSION - 1)).1.0/bin/g++ /usr/bin/g++-$((GCC_HEAD_VERSION - 1))
  # For CI transitions.
  sudo /tmp/ce-infra/bin/ce_install install compilers/c++/x86/gcc $((GCC_HEAD_VERSION - 2)).1.0
  sudo ln -s /opt/compiler-explorer/gcc-$((GCC_HEAD_VERSION - 2)).1.0/bin/gcc /usr/bin/gcc-$((GCC_HEAD_VERSION - 2))
  sudo ln -s /opt/compiler-explorer/gcc-$((GCC_HEAD_VERSION - 2)).1.0/bin/g++ /usr/bin/g++-$((GCC_HEAD_VERSION - 2))
  sudo rm -rf /tmp/ce-infra
EOF

RUN <<EOF
    # Install a recent CMake
    set -e
    wget https://github.com/Kitware/CMake/releases/download/v3.24.4/cmake-3.24.4-linux-x86_64.sh -O /tmp/install-cmake.sh
    sudo bash /tmp/install-cmake.sh --prefix=/usr --exclude-subdir --skip-license
    rm /tmp/install-cmake.sh
EOF
