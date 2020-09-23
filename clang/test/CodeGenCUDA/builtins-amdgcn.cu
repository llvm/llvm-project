// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -emit-llvm %s -o - | FileCheck %s
#include "Inputs/cuda.h"

// CHECK-LABEL: @_Z16use_dispatch_ptrPi(
// CHECK: %[[PTR:.*]] = call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
// CHECK: %{{.*}} = addrspacecast i8 addrspace(4)* %[[PTR]] to i8*
__global__ void use_dispatch_ptr(int* out) {
  const int* dispatch_ptr = (const int*)__builtin_amdgcn_dispatch_ptr();
  *out = *dispatch_ptr;
}

// CHECK-LABEL: @_Z12test_ds_fmaxf(
// CHECK: call contract float @llvm.amdgcn.ds.fmax.f32(float addrspace(3)* @_ZZ12test_ds_fmaxfE6shared, float %{{[^,]*}}, i32 0, i32 0, i1 false)
__global__
void test_ds_fmax(float src) {
  __shared__ float shared;
  volatile float x = __builtin_amdgcn_ds_fmaxf(&shared, src, 0, 0, false);
}
