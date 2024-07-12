FROM docker.io/library/ubuntu:22.04 as base
ENV LLVM_SYSROOT=/opt/llvm

FROM base as stage1-toolchain
ENV LLVM_VERSION=17.0.6

RUN apt-get update && \
    apt-get install -y \
    wget \
    gcc \
    g++ \
    cmake \
    ninja-build \
    python3 \
    git \
    curl

RUN curl -O -L https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-$LLVM_VERSION.tar.gz && tar -xf llvmorg-$LLVM_VERSION.tar.gz

WORKDIR /llvm-project-llvmorg-$LLVM_VERSION

COPY bootstrap.patch /

# TODO(boomanaiden154): Remove the patch pulled from a LLVM PR once we bump
# the toolchain to version 18 and the patch is in-tree.
# TODO(boomanaiden154): Remove the bootstrap patch once we unsplit the build
# and no longer need to explicitly build the stage2 dependencies.
RUN curl https://github.com/llvm/llvm-project/commit/dd0356d741aefa25ece973d6cc4b55dcb73b84b4.patch | patch -p1 && cat /bootstrap.patch | patch -p1

RUN mkdir build

RUN cmake -B ./build -G Ninja ./llvm \
  -C ./clang/cmake/caches/BOLT-PGO.cmake \
  -DBOOTSTRAP_LLVM_ENABLE_LLD=ON \
  -DBOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_LLD=ON \
  -DPGO_INSTRUMENT_LTO=Thin \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_INSTALL_PREFIX="$LLVM_SYSROOT" \
  -DLLVM_ENABLE_PROJECTS="bolt;clang;lld;clang-tools-extra" \
  -DLLVM_DISTRIBUTION_COMPONENTS="lld;compiler-rt;clang-format;scan-build" \
  -DCLANG_DEFAULT_LINKER="lld" \
  -DBOOTSTRAP_CLANG_PGO_TRAINING_DATA_SOURCE_DIR=/llvm-project-llvmorg-$LLVM_VERSION/llvm

RUN ninja -C ./build stage2-instrumented-clang stage2-instrumented-lld
