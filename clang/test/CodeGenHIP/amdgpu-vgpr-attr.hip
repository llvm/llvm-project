// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -fcuda-is-device -emit-llvm -O1 -disable-llvm-passes -o - %s \
// RUN:   | FileCheck %s

#define __global__ __attribute__((global))

// A kernel-local variable marked amdgpu_vgpr is allocated in the "VGPR as
// memory" address space (addrspace(13)), and its accesses stay in that space.

// CHECK-LABEL: define {{.*}}@_Z6kernelPii(
// CHECK: %buf = alloca [4 x i32], align 4, addrspace(13)
// CHECK: getelementptr inbounds [4 x i32], ptr addrspace(13) %buf
// CHECK: store i32 %{{.*}}, ptr addrspace(13)
// CHECK: load i32, ptr addrspace(13)
__global__ void kernel(int *out, int i) {
  int buf[4] __attribute__((amdgpu_vgpr));
  buf[2] = i;
  out[0] = buf[2];
}
