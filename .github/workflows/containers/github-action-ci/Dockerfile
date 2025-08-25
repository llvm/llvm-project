FROM docker.io/library/ubuntu:24.04 as base
ENV LLVM_SYSROOT=/opt/llvm

FROM base as stage1-toolchain
ENV LLVM_VERSION=20.1.8

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
  -C ./clang/cmake/caches/BOLT-PGO.cmake \
  -DBOOTSTRAP_LLVM_ENABLE_LLD=ON \
  -DBOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_LLD=ON \
  -DPGO_INSTRUMENT_LTO=Thin \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_INSTALL_PREFIX="$LLVM_SYSROOT" \
  -DLLVM_ENABLE_PROJECTS="bolt;clang;lld;clang-tools-extra" \
  -DLLVM_DISTRIBUTION_COMPONENTS="lld;compiler-rt;clang-format;scan-build;llvm-symbolizer" \
  -DCLANG_DEFAULT_LINKER="lld"

RUN ninja -C ./build stage2-clang-bolt stage2-install-distribution && ninja -C ./build install-distribution

FROM base as ci-container

COPY --from=stage1-toolchain $LLVM_SYSROOT $LLVM_SYSROOT

# Need to install curl for hendrikmuhs/ccache-action
# Need nodejs for some of the GitHub actions.
# Need perl-modules for clang analyzer tests.
# Need git for SPIRV-Tools tests.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    binutils \
    cmake \
    curl \
    git \
    libstdc++-11-dev \
    ninja-build \
    nodejs \
    perl-modules \
    python3-psutil \
    sudo \
    # These are needed by the premerge pipeline. Pip is used to install
    # dependent python packages. File and tzdata are used for tests.
    # Having a symlink from python to python3 enables code sharing between
    # the Linux and Windows pipelines.
    python3-pip \
    file \
    tzdata \
    python-is-python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# We need sccache for caching. We cannot use the apt repository version because
# it is too old and has bugs related to features we require (particularly GCS
# caching), so we manually install it here.
# TODO(boomanaiden154): We should return to installing this from the apt
# repository once a version containing the necessary bug fixes is available.
RUN curl -L 'https://github.com/mozilla/sccache/releases/download/v0.10.0/sccache-v0.10.0-x86_64-unknown-linux-musl.tar.gz' > /tmp/sccache.tar.gz && \
    echo "1fbb35e135660d04a2d5e42b59c7874d39b3deb17de56330b25b713ec59f849b /tmp/sccache.tar.gz" | sha256sum -c && \
    tar xzf /tmp/sccache.tar.gz -O --wildcards '*/sccache' > '/usr/local/bin/sccache' && \
    rm /tmp/sccache.tar.gz && \
    chmod +x /usr/local/bin/sccache

ENV LLVM_SYSROOT=$LLVM_SYSROOT
ENV PATH=${LLVM_SYSROOT}/bin:${PATH}
ENV CC=clang
ENV CXX=clang++

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

FROM ci-container as ci-container-agent

ENV GITHUB_RUNNER_VERSION=2.328.0

RUN mkdir actions-runner && \
    cd actions-runner && \
    curl -O -L https://github.com/actions/runner/releases/download/v$GITHUB_RUNNER_VERSION/actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz && \
    tar xzf ./actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz && \
    rm ./actions-runner-linux-x64-$GITHUB_RUNNER_VERSION.tar.gz

