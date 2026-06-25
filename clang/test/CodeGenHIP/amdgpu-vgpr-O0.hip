// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -fcuda-is-device -O0 -emit-llvm -verify -o - %s | FileCheck %s
//
// "VGPR as memory" is honored at every optimization level (it is a global, not
// an alloca that depends on the optimizing register allocator), so at -O0 the
// variable is still emitted in addrspace(13) with no diagnostic.

// expected-no-diagnostics

#define __global__ __attribute__((global))

// CHECK: @{{.*}}buf = internal addrspace(13) global [4 x i32] poison, align 4
// CHECK-LABEL: define {{.*}}@_Z6kernelPii(
// CHECK: store i32 %{{.*}}, ptr addrspace(13) {{.*}}@{{.*}}buf
__global__ void kernel(int *out, int i) {
  int buf[4] __attribute__((amdgpu_vgpr));
  buf[2] = i;
  out[0] = buf[2];
}
