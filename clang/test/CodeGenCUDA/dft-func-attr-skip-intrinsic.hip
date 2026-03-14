// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x ir -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm-bc -disable-llvm-passes -o %t.bc %S/Inputs/device-lib-code.ll
// RUN: %clang_cc1 -x hip -fcuda-is-device -triple amdgcn-amd-amdhsa -mlink-builtin-bitcode %t.bc -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

#include "Inputs/cuda.h"

extern "C" __device__ float __ocml_fma_f32(float x, float y, float z);

__device__ float foo(float x) {
  return __ocml_fma_f32(x, x, x);
}

// CHECK: {{^}}define{{.*}} @__ocml_fma_f32{{.*}} [[ATTR1:#[0-9]+]]
// CHECK: {{^}}declare{{.*}} @llvm.fma.f32{{.*}} [[ATTR2:#[0-9]+]]
// CHECK: attributes [[ATTR1]] = { convergent
// CHECK: attributes [[ATTR2]] = {
// CHECK-NOT: convergent
// CHECK-SAME: }
