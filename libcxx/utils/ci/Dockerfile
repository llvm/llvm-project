#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This Dockerfile describes the base image used to run the various libc++
# build bots. By default, the image runs the Buildkite Agent, however one
# can also just start the image with a shell to debug CI failures.
#
# To start a Buildkite Agent, run it as:
#   $ docker run --env-file <secrets> -it $(docker build -q .)
#
# The environment variables in `<secrets>` should be the ones necessary
# to run a BuildKite agent.
#
# If you're only looking to run the Docker image locally for debugging a
# build bot, see the `run-buildbot-container` script located in this directory.
#
# A pre-built version of this image is maintained on DockerHub as ldionne/libcxx-builder.
# To update the image, rebuild it and push it to ldionne/libcxx-builder (which
# will obviously only work if you have permission to do so).
#
#   $ docker build -t ldionne/libcxx-builder .
#   $ docker push ldionne/libcxx-builder
#

FROM ubuntu:bionic

# Make sure apt-get doesn't try to prompt for stuff like our time zone, etc.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y bash curl

# Install various tools used by the build or the test suite
RUN apt-get update && apt-get install -y ninja-build python3 python3-sphinx python3-distutils git gdb
RUN apt-get update && apt-get install -y libc6-dev-i386 # Required to cross-compile to 32 bits

# Install Clang <latest>, <latest-1> and ToT, which are the ones we support.
ENV LLVM_LATEST_VERSION=12
RUN apt-get update && apt-get install -y lsb-release wget software-properties-common
RUN wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
RUN bash /tmp/llvm.sh $(($LLVM_LATEST_VERSION - 1)) # previous release
RUN bash /tmp/llvm.sh $LLVM_LATEST_VERSION          # latest release
RUN bash /tmp/llvm.sh $(($LLVM_LATEST_VERSION + 1)) # current ToT

# Make the ToT Clang the "default" compiler on the system
RUN ln -fs /usr/bin/clang++-$(($LLVM_LATEST_VERSION + 1)) /usr/bin/c++ && [ -e $(readlink /usr/bin/c++) ]
RUN ln -fs /usr/bin/clang-$(($LLVM_LATEST_VERSION + 1)) /usr/bin/cc && [ -e $(readlink /usr/bin/cc) ]

# Install clang-format
RUN apt-get install -y clang-format-$LLVM_LATEST_VERSION
RUN ln -s /usr/bin/clang-format-$LLVM_LATEST_VERSION /usr/bin/clang-format && [ -e $(readlink /usr/bin/clang-format) ]
RUN ln -s /usr/bin/git-clang-format-$LLVM_LATEST_VERSION /usr/bin/git-clang-format && [ -e $(readlink /usr/bin/git-clang-format) ]

# Install the most recent GCC
ENV GCC_LATEST_VERSION=11
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt install -y gcc-$GCC_LATEST_VERSION g++-$GCC_LATEST_VERSION

# Install a recent CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh -O /tmp/install-cmake.sh
RUN bash /tmp/install-cmake.sh --prefix=/usr --exclude-subdir --skip-license
RUN rm /tmp/install-cmake.sh

# Change the user to a non-root user, since some of the libc++ tests
# (e.g. filesystem) require running as non-root. Also setup passwordless sudo.
RUN apt-get update && apt-get install -y sudo
RUN echo "ALL ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN useradd --create-home libcxx-builder
USER libcxx-builder
WORKDIR /home/libcxx-builder

# Install the Buildkite agent and dependencies. This must be done as non-root
# for the Buildkite agent to be installed in a path where we can find it.
RUN bash -c "$(curl -sL https://raw.githubusercontent.com/buildkite/agent/master/install.sh)"
ENV PATH="${PATH}:/home/libcxx-builder/.buildkite-agent/bin"
RUN echo "tags=\"queue=libcxx-builders,arch=$(uname -m),os=linux\"" >> "/home/libcxx-builder/.buildkite-agent/buildkite-agent.cfg"

# By default, start the Buildkite agent (this requires a token).
CMD buildkite-agent start
