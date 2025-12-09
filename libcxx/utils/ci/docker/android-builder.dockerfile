# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
#
# This file defines the image we use to run Android testing on Buildkite.
# From the root of the monorepo, this image can be built with:
#
#   $ docker build --file libcxx/utils/ci/docker/android-builder.dockerfile \
#                --build-arg BASE_IMAGE_VERSION=<sha>                       \
#                --build-arg ANDROID_CLANG_VERSION=<version>                \
#                --build-arg ANDROID_CLANG_PREBUILTS_COMMIT=<sha>           \
#                --build-arg ANDROID_SYSROOT_COMMIT=<sha> .
#
# This image also gets built on every push to `main` that modifies these Docker
# files, and can be found at ghcr.io/llvm/libcxx-android-builder.
#
# To run the image and start a Buildkite Agent, run it as:
#
#   $ docker run --env-file <secrets> -it ghcr.io/llvm/libcxx-android-builder:latest
#
# The environment variables in `<secrets>` should be the ones necessary
# to run a BuildKite agent:
#
#   BUILDKITE_AGENT_TOKEN=<token>

ARG BASE_IMAGE_VERSION
FROM ghcr.io/llvm/libcxx-linux-builder-base:${BASE_IMAGE_VERSION}

ARG ANDROID_CLANG_VERSION
ARG ANDROID_CLANG_PREBUILTS_COMMIT
ARG ANDROID_SYSROOT_COMMIT

# Install the Android platform tools (e.g. adb) into /opt/android/sdk.
RUN <<EOF
  set -e
  mkdir -p /opt/android/sdk
  cd /opt/android/sdk
  curl -LO https://dl.google.com/android/repository/platform-tools-latest-linux.zip
  unzip platform-tools-latest-linux.zip
  rm platform-tools-latest-linux.zip
EOF

# Install the current Android compiler. Specify the prebuilts commit to retrieve
# this compiler version even after it's removed from HEAD.
ENV ANDROID_CLANG_VERSION=$ANDROID_CLANG_VERSION
ENV ANDROID_CLANG_PREBUILTS_COMMIT=$ANDROID_CLANG_PREBUILTS_COMMIT
RUN <<EOF
    set -e
    git clone --filter=blob:none --sparse \
        https://android.googlesource.com/platform/prebuilts/clang/host/linux-x86 \
        /opt/android/clang
    git -C /opt/android/clang checkout ${ANDROID_CLANG_PREBUILTS_COMMIT}
    git -C /opt/android/clang sparse-checkout add clang-${ANDROID_CLANG_VERSION}
    rm -fr /opt/android/clang/.git
    ln -sf /opt/android/clang/clang-${ANDROID_CLANG_VERSION} /opt/android/clang/clang-current
    # The "git sparse-checkout" and "ln" commands succeed even if nothing was
    # checked out, so use this "ls" command to fix that.
    ls /opt/android/clang/clang-current/bin/clang
EOF

# Install an Android sysroot. New Android sysroots are available at
# https://android.googlesource.com/platform/prebuilts/ndk/+/refs/heads/mirror-goog-main-ndk/platform/sysroot.
ENV ANDROID_SYSROOT_COMMIT=$ANDROID_SYSROOT_COMMIT
RUN <<EOF
  set -e
  mkdir -p /opt/android/ndk
  cd /opt/android/ndk
  git clone --filter=blob:none https://android.googlesource.com/platform/prebuilts/ndk tmp
  git -C tmp checkout ${ANDROID_SYSROOT_COMMIT}
  mv tmp/platform/sysroot .
  rm -rf tmp
EOF

# Create the libcxx-builder user
RUN sudo useradd --create-home libcxx-builder
WORKDIR /home/libcxx-builder

# Install the Buildkite agent and dependencies. This must be done as non-root
# for the Buildkite agent to be installed in a path where we can find it.
#
# Note that we don't set up the configuration file here, since the configuration
# is passed via the environment.
RUN <<EOF
  set -e
  cd /home/libcxx-builder
  curl -sL https://raw.githubusercontent.com/buildkite/agent/main/install.sh -o /tmp/install-agent.sh
  bash /tmp/install-agent.sh
  rm /tmp/install-agent.sh
EOF
ENV PATH="${PATH}:/home/libcxx-builder/.buildkite-agent/bin"

# Install Docker
RUN <<EOF
  set -e
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sh /tmp/get-docker.sh
  rm /tmp/get-docker.sh

  # Install Docker. Mark the binary setuid so it can be run without prefixing it
  # with sudo. Adding the container user to the docker group doesn't work because
  # /var/run/docker.sock is owned by the host's docker GID, not the container's
  # docker GID.
  chmod u+s /usr/bin/docker
EOF

# Setup the Buildkite agent command line to do Android setup stuff first.
USER libcxx-builder
COPY libcxx/utils/ci/vendor/android/container-setup.sh /opt/android/container-setup.sh
ENV PATH="/opt/android/sdk/platform-tools:${PATH}"
CMD /opt/android/container-setup.sh && buildkite-agent start
