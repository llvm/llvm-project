// REQUIRES: amdgpu-registered-target
// REQUIRES: nvptx-registered-target

// Implicit-H+D trivial dtor of an explicit template instantiation: vtable
// slot uses the real mangled name, dtor body is real. With or without an
// organic device caller, the emitted IR is the same.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -DWITH_DEVICE_CALLER -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -x hip \
// RUN:   -DWITH_DEVICE_CALLER -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

template <typename T>
struct ETI {
  virtual ~ETI() = default;
};
template class ETI<float>;

#ifdef WITH_DEVICE_CALLER
__device__ void use(ETI<float> *p) { p->~ETI(); }
__global__ void kernel(ETI<float> *p) { use(p); }
#endif

// CHECK: @_ZTV3ETIIfE =
// CHECK-SAME: @_ZN3ETIIfED1Ev
// CHECK-SAME: @_ZN3ETIIfED0Ev
// CHECK: define {{.*}} @_ZN3ETIIfED1Ev(
// CHECK-NOT: call void @llvm.trap()
// CHECK: ret void
// CHECK: define {{.*}} @_ZN3ETIIfED0Ev(
// CHECK-NOT: call void @llvm.trap()
// CHECK: ret void
// CHECK-NOT: __clang_hip_unreachable_dtor
