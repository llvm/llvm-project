// REQUIRES: amdgpu-registered-target
// REQUIRES: nvptx-registered-target

// Overload ambiguity inside an implicit-H+D explicit-inst member without an
// organic device caller: error is silenced, function body is a trap stub.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

__host__ __device__ int pick(long);
__host__ __device__ int pick(unsigned long);

template <typename T>
struct ETI {
  constexpr int call(T x) { return pick(x); }
};
template class ETI<int>;

__global__ void k() {}

// CHECK-LABEL: define {{.*}} i32 @_ZN3ETIIiE4callEi(
// CHECK: call void @llvm.trap()
// CHECK-NEXT: ret i32 poison
