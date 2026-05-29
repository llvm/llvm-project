// REQUIRES: amdgpu-registered-target
// REQUIRES: nvptx-registered-target

// Implicit-H+D functions reached only via an explicit template instantiation
// get a real body if their device codegen is clean, or a trap body if it
// would otherwise raise a target-mismatch diagnostic.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

__host__ int host_only() { return 7; }

// Class-template member, safe body.
template <typename T>
struct ETI_Safe {
  constexpr T add_one(T x) { return x + (T)1; }
};
template class ETI_Safe<float>;

// CHECK: define {{.*}} float @_ZN8ETI_SafeIfE7add_oneEf(
// CHECK-NOT: call void @llvm.trap()
// CHECK: ret float

// Class-template member, body calls a __host__-only function.
template <typename T>
struct ETI_BadBody {
  constexpr T call_host(T x) { return x + (T)host_only(); }
};
template class ETI_BadBody<float>;

// CHECK-LABEL: define {{.*}} float @_ZN11ETI_BadBodyIfE9call_hostEf(
// CHECK: call void @llvm.trap()
// CHECK-NEXT: ret float poison

// Function-template form, safe body.
template <typename T>
constexpr T ft_safe(T x) { return x + (T)1; }
template float ft_safe<float>(float);

// CHECK: define {{.*}} float @_Z7ft_safeIfET_S0_(
// CHECK-NOT: call void @llvm.trap()
// CHECK: ret float

// Function-template form, body calls a __host__-only function.
template <typename T>
constexpr T ft_bad(T x) { return x + (T)host_only(); }
template float ft_bad<float>(float);

// CHECK-LABEL: define {{.*}} float @_Z6ft_badIfET_S0_(
// CHECK: call void @llvm.trap()
// CHECK-NEXT: ret float poison

__global__ void kernel(float *p) { *p = 1.0f; }

// CHECK-NOT: __clang_hip_unreachable_dtor
// CHECK-NOT: __clang_cuda_unreachable_dtor
