FROM docker.io/library/ubuntu:24.04 AS builder

# Install perf as it is missing from 24.04 packages:
# https://bugs.launchpad.net/ubuntu/+source/linux-hwe-6.14/+bug/2117147

ARG PERF_VER=6.17
RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    ca-certificates wget xz-utils \
    libtraceevent-dev libtracefs-dev \
    build-essential flex bison \
    libelf-dev libdw-dev zlib1g-dev liblzma-dev libcap-dev libnuma-dev \
    python3 python3-dev python3-setuptools \
    pkg-config; \
  rm -rf /var/lib/apt/lists/*; \
  wget -O /tmp/perf.tar.xz "https://mirrors.edge.kernel.org/pub/linux/kernel/tools/perf/v${PERF_VER}.0/perf-${PERF_VER}.0.tar.xz"; \
  mkdir -p /src && tar -C /src -xf /tmp/perf.tar.xz; \
  cd "/src/perf-${PERF_VER}.0"; \
  make -C tools/perf -j"$(nproc)"; \
  install -m 0755 tools/perf/perf /usr/local/bin/perf; \
  /usr/local/bin/perf --version >&2

# Install tools for BOLT compilation.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git \
      build-essential cmake ninja-build python3 zstd ccache \
      python3-psutil && \
    rm -rf /var/lib/apt/lists

WORKDIR /home/bolt

# Get sources, compile BOLT, and run test suites.
RUN git clone --depth 1 https://github.com/llvm/llvm-project
RUN git clone --depth 1 https://github.com/rafaelauler/bolt-tests

RUN mkdir build && \
    cd build && \
    cmake -G Ninja ../llvm-project/llvm \
      -DLLVM_ENABLE_PROJECTS="bolt;clang;lld" \
      -DLLVM_TARGETS_TO_BUILD="AArch64;X86" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_CCACHE_BUILD=ON \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_EXTERNAL_PROJECTS="bolttests" \
      -DLLVM_EXTERNAL_BOLTTESTS_SOURCE_DIR=../bolt-tests

RUN cd build && ninja check-bolt
RUN cd build && ninja check-large-bolt
