#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

#
# This Dockerfile extends ldionne/libcxx-builder with Android support, including
# Android Clang and sysroot, Android platform-tools, and the Docker client.
#
#   $ docker build -t libcxx-builder-android libcxx/utils/ci/vendor/android
#

FROM ldionne/libcxx-builder

# Switch back to the root user to install things into /opt and /usr.
USER root
WORKDIR /

# Install the Android platform tools (e.g. adb) into /opt/android/sdk.
RUN apt-get update && apt-get install -y unzip
RUN mkdir -p /opt/android/sdk && cd /opt/android/sdk && \
    curl -LO https://dl.google.com/android/repository/platform-tools-latest-linux.zip && \
    unzip platform-tools-latest-linux.zip && \
    rm platform-tools-latest-linux.zip

# Install the current Android compiler. Specify the prebuilts commit to retrieve
# this compiler version even after it's removed from HEAD.
ENV ANDROID_CLANG_VERSION=r498229b
ENV ANDROID_CLANG_PREBUILTS_COMMIT=5186d132c99aa75dc25207c392e3ea5b93d0107e
RUN git clone --filter=blob:none --sparse \
        https://android.googlesource.com/platform/prebuilts/clang/host/linux-x86 \
        /opt/android/clang && \
    git -C /opt/android/clang checkout ${ANDROID_CLANG_PREBUILTS_COMMIT} && \
    git -C /opt/android/clang sparse-checkout add clang-${ANDROID_CLANG_VERSION} && \
    rm -fr /opt/android/clang/.git && \
    ln -sf /opt/android/clang/clang-${ANDROID_CLANG_VERSION} /opt/android/clang/clang-current && \
    # The "git sparse-checkout" and "ln" commands succeed even if nothing was
    # checked out, so use this "ls" command to fix that.
    ls /opt/android/clang/clang-current/bin/clang

# Install an Android sysroot. New AOSP sysroots are available at
# https://ci.android.com/builds/branches/aosp-main/grid, the "ndk" target. The
# NDK also makes its sysroot prebuilt available at
# https://android.googlesource.com/platform/prebuilts/ndk/+/refs/heads/dev/platform/sysroot.
ENV ANDROID_SYSROOT_BID=10957860
RUN cd /opt/android && \
    curl -L -o ndk_platform.tar.bz2 \
        https://androidbuildinternal.googleapis.com/android/internal/build/v3/builds/${ANDROID_SYSROOT_BID}/ndk/attempts/latest/artifacts/ndk_platform.tar.bz2/url && \
    tar xf ndk_platform.tar.bz2 && \
    rm ndk_platform.tar.bz2

# Install Docker. Mark the binary setuid so it can be run without prefixing it
# with sudo. Adding the container user to the docker group doesn't work because
# /var/run/docker.sock is owned by the host's docker GID, not the container's
# docker GID.
RUN apt-get update && apt-get install -y gpg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
         "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
         tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    chmod u+s /usr/bin/docker

COPY ./container-setup.sh /opt/android/container-setup.sh

USER libcxx-builder
WORKDIR /home/libcxx-builder

# Add Android platform-tools to the PATH.
ENV PATH="/opt/android/sdk/platform-tools:${PATH}"

# Reset the buildkite-agent.cfg file. The tags are provided by an environment
# variable instead.
RUN cp /home/libcxx-builder/.buildkite-agent/buildkite-agent.dist.cfg \
       /home/libcxx-builder/.buildkite-agent/buildkite-agent.cfg

# Modify the Buildkite agent cmdline to do Android setup stuff first.
CMD /opt/android/container-setup.sh && buildkite-agent start
