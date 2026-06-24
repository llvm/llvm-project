// RUN: not %clangxx -### --target=x86_64-linux-gnu -S -emit-llvm --cuda-gpu-arch=sm_52 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 | FileCheck %s
//
// CHECK: error: cannot create CUDA fatbinary from LLVM IR/bitcode
// CHECK-NOT: "--create"

__global__ void k() {}
