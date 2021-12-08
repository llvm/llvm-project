// RUN: %clang_cc1 -triple spirv64 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

#define __global__ __attribute__((global))

// CHECK: define {{.*}}spir_kernel void @_Z3fooPff(float addrspace(1)* {{.*}}, float {{.*}})
__global__ void foo(float *a, float b) {
  *a = b;
}
