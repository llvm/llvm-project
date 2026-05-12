// REQUIRES: amdgpu-registered-target

// HIP device pass uses __clang_hip_ (not __clang_cuda_) for the trap
// stub name.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include "Inputs/cuda.h"

template <typename T>
struct ETI {
  virtual ~ETI() = default;
};
template class ETI<float>;

// CHECK: @_ZTV3ETIIfE =
// CHECK-SAME: @__clang_hip_unreachable_dtor._ZN3ETIIfED1Ev
// CHECK-SAME: @__clang_hip_unreachable_dtor._ZN3ETIIfED0Ev
// CHECK: define internal void @__clang_hip_unreachable_dtor._ZN3ETIIfED1Ev
// CHECK: call void @llvm.trap()
// CHECK-NEXT: unreachable
