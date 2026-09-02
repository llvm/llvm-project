#!/bin/sh

set -eux

cmake -B build.$1 \
      --toolchain $2 \
      -C cmake/caches/O3.cmake \
      -GNinja \
      -DTEST_SUITE_BENCHMARKING_ONLY=ON \
      -DTEST_SUITE_RUN_BENCHMARKS=OFF
ninja -C build.$1
llvm-lit build.$1 -o results.$1.json
