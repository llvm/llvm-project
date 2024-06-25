// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __device__ __attribute__((device))

// CHECK-LABEL: @_Z2d0DF16_
// CHECK: fpext
__device__ float d0(_Float16 x) {
  return x;
}

// CHECK-LABEL: @_Z2d1f
// CHECK: fptrunc
__device__ _Float16 d1(float x) {
  return x;
}
