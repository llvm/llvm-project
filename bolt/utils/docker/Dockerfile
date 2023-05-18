FROM ubuntu:20.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git \
      build-essential cmake ninja-build python3 libjemalloc-dev \
      python3-psutil && \
    rm -rf /var/lib/apt/lists

WORKDIR /home/bolt

RUN git clone --depth 1 https://github.com/llvm/llvm-project

RUN mkdir build && \
    cd build && \
    cmake -G Ninja ../llvm-project/llvm \
      -DLLVM_ENABLE_PROJECTS="bolt;clang;lld" \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,--push-state -Wl,-whole-archive -ljemalloc_pic -Wl,--pop-state -lpthread -lstdc++ -lm -ldl" \
      -DCMAKE_INSTALL_PREFIX=/home/bolt/install && \
    ninja check-bolt && \
    ninja install-llvm-bolt install-perf2bolt install-merge-fdata \
      install-llvm-boltdiff install-bolt_rt

FROM ubuntu:20.04

COPY --from=builder /home/bolt/install /usr/local
