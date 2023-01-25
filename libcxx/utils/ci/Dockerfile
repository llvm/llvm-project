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
#   $ docker run --env-file <secrets> -it $(docker build -q libcxx/utils/ci)
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
#   $ docker build -t ldionne/libcxx-builder libcxx/utils/ci
#   $ docker push ldionne/libcxx-builder
#

FROM ubuntu:jammy

# Make sure apt-get doesn't try to prompt for stuff like our time zone, etc.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y bash curl

# Install various tools used by the build or the test suite
RUN apt-get update && apt-get install -y ninja-build python3 python3-sphinx python3-distutils python3-psutil git gdb

# Install dependencies required to run the LLDB data formatter tests
RUN apt-get update && apt-get install -y python3 python3-dev libpython3-dev uuid-dev libncurses5-dev swig3.0 libxml2-dev libedit-dev

# Locales for gdb and localization tests
RUN apt-get update && apt-get install -y language-pack-en language-pack-fr \
                                         language-pack-ja language-pack-ru \
                                         language-pack-zh-hans
# These two are not enabled by default so generate them
RUN printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2" >> /etc/locale.gen
RUN mkdir /usr/local/share/i1en/
RUN printf "fr_CA ISO-8859-1\ncs_CZ ISO-8859-2" >> /usr/local/share/i1en/SUPPORTED
RUN locale-gen

# Install Clang <latest>, <latest-1> and ToT, which are the ones we support.
# We also install <latest-2> because we need to support the "latest-1" of the
# current LLVM release branch, which is effectively the <latest-2> of the
# tip-of-trunk LLVM. For example, after branching LLVM 14 but before branching
# LLVM 15, we still need to have Clang 12 in this Docker image because the LLVM
# 14 release branch CI uses it. The tip-of-trunk CI will never use Clang 12,
# though.
ENV LLVM_HEAD_VERSION=16
RUN apt-get update && apt-get install -y lsb-release wget software-properties-common
RUN wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
RUN bash /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 3)) # for CI transitions
RUN bash /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 2)) # previous release
RUN bash /tmp/llvm.sh $(($LLVM_HEAD_VERSION - 1)) # latest release
RUN bash /tmp/llvm.sh $LLVM_HEAD_VERSION          # current ToT

# Make the latest version of Clang the "default" compiler on the system
# TODO: After branching for LLVM 16, all jobs will be using an
#       explicitly-versioned version of Clang instead, so we can get rid of
#       these symlinks.
RUN ln -fs /usr/bin/clang++-14 /usr/bin/c++ && [ -e $(readlink /usr/bin/c++) ]
RUN ln -fs /usr/bin/clang-14 /usr/bin/cc && [ -e $(readlink /usr/bin/cc) ]

# Install clang-format; always use the lastest stable branch.
# TODO LLVM 16 remove hard-coded version 14 and the symlinks
RUN apt-get update && apt-get install -y clang-format-$(($LLVM_HEAD_VERSION - 2)) clang-format-$(($LLVM_HEAD_VERSION - 1))
RUN ln -s /usr/bin/clang-format-14 /usr/bin/clang-format && [ -e $(readlink /usr/bin/clang-format) ]
RUN ln -s /usr/bin/git-clang-format-14 /usr/bin/git-clang-format && [ -e $(readlink /usr/bin/git-clang-format) ]

# Install clang-tidy
# TODO LLVM 16 remove hard-coded version 14 and the symlink
RUN apt-get update && apt-get install -y clang-tidy-$(($LLVM_HEAD_VERSION - 1)) clang-tidy-$LLVM_HEAD_VERSION clang-tidy-14
RUN ln -s /usr/bin/clang-tidy-14 /usr/bin/clang-tidy && [ -e $(readlink /usr/bin/clang-tidy) ]

# Install llvm-dev and libclang-dev to compile custom clang-tidy checks
RUN apt-get update && apt-get install -y llvm-$(($LLVM_HEAD_VERSION - 1))-dev     llvm-$(($LLVM_HEAD_VERSION - 2))-dev \
                                     libclang-$(($LLVM_HEAD_VERSION - 1))-dev libclang-$(($LLVM_HEAD_VERSION - 2))-dev

# TODO LLVM16 Don't install llvm-16-dev explicitly
RUN apt-get update && apt-get install -y llvm-16-dev libclang-16-dev

# Install the most recent GCC, like clang install the previous version as a transition.
ENV GCC_LATEST_VERSION=12
RUN apt-get update && apt install -y gcc-$((GCC_LATEST_VERSION - 1)) g++-$((GCC_LATEST_VERSION - 1))
RUN apt-get update && apt install -y gcc-$GCC_LATEST_VERSION g++-$GCC_LATEST_VERSION

# Install a recent CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.sh -O /tmp/install-cmake.sh
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
RUN bash -c "$(curl -sL https://raw.githubusercontent.com/buildkite/agent/main/install.sh)"
ENV PATH="${PATH}:/home/libcxx-builder/.buildkite-agent/bin"
RUN echo "tags=\"queue=libcxx-builders,arch=$(uname -m),os=linux\"" >> "/home/libcxx-builder/.buildkite-agent/buildkite-agent.cfg"

# By default, start the Buildkite agent (this requires a token).
CMD buildkite-agent start
