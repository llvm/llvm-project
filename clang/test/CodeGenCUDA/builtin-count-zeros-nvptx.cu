// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -x cuda -triple nvptx64-unknown-unknown -fcuda-is-device -emit-llvm %s -o - | FileCheck %s
//
// Ensure NVPTX uses isCLZForZeroUndef() = false (CUDA semantics: CLZ(i32 0) == 32).

#include "Inputs/cuda.h"

__device__ int f(int x) {
  return __builtin_ctz(x) + __builtin_clz(x);
}
// CHECK: call i32 @llvm.cttz.i32({{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32({{.*}}, i1 false)
