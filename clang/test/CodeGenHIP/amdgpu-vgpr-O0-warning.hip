// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -fcuda-is-device -O0 -emit-llvm -verify -o - %s | FileCheck %s
//
// At -O0 "VGPR as memory" is not honored: the variable falls back to an
// ordinary (scratch) alloca in addrspace(5) and a warning is emitted.

#define __global__ __attribute__((global))

// CHECK: %buf = alloca [4 x i32], align 4, addrspace(5)
__global__ void kernel(int *out, int i) {
  int buf[4] __attribute__((amdgpu_vgpr)); // expected-warning {{'amdgpu_vgpr' is not guaranteed to keep the variable in vector registers at -O0; it may fall back to scratch memory}}
  buf[2] = i;
  out[0] = buf[2];
}
