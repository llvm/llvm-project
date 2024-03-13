#!/bin/bash

# 直接 release 版本

mkdir build-release && cd build-release
cmake -G Ninja \
      -DLLVM_ENABLE_PROJECTS='clang' \
      -DCMAKE_BUILD_TYPE=Release \
      ../llvm

# 打开对 typeid 的支持（感觉没用）
      # -DLLVM_ENABLE_RTTI=true \

#      -DLLVM_PARALLEL_COMPILE_JOBS=12 \
