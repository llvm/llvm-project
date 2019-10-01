// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility default -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility protected -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-PROTECTED %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility hidden -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-HIDDEN %s

#include "Inputs/cuda.h"

// CHECK-DEFAULT: @c = addrspace(4) externally_initialized global
// CHECK-DEFAULT: @g = addrspace(1) externally_initialized global
// CHECK-PROTECTED: @c = protected addrspace(4) externally_initialized global
// CHECK-PROTECTED: @g = protected addrspace(1) externally_initialized global
// CHECK-HIDDEN: @c = protected addrspace(4) externally_initialized global
// CHECK-HIDDEN: @g = protected addrspace(1) externally_initialized global
__constant__ int c;
__device__ int g;

// CHECK-DEFAULT: define amdgpu_kernel void @_Z3foov()
// CHECK-PROTECTED: define protected amdgpu_kernel void @_Z3foov()
// CHECK-HIDDEN: define protected amdgpu_kernel void @_Z3foov()
__global__ void foo() {
  g = c;
}
