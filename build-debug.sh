#!/bin/bash

# 用于 debug
# 最终结果会很大，buid文件夹大约100G左右
# 使用 LLVM_PARALLEL_LINK_JOBS 限制并行链接的个数，防止爆内存
mkdir build-debug && cd build-debug
cmake -G Ninja \
      -DLLVM_ENABLE_PROJECTS='clang' \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_PARALLEL_COMPILE_JOBS=12 \
      -DLLVM_PARALLEL_LINK_JOBS=2 \
      ../llvm
