// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 \
// RUN:   -fcuda-is-device -emit-llvm -O1 -disable-llvm-passes -o - %s \
// RUN:   | FileCheck %s

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

// A variable marked amdgpu_vgpr is emitted as an internal global in the "VGPR
// as memory" address space (addrspace(13)) with a poison initializer (like an
// LDS/__shared__ variable), and its accesses stay in that space. It is allowed
// in a __device__ function too, not just a __global__ kernel.

// CHECK-DAG: @{{.*}}kernel{{.*}} = internal addrspace(13) global [4 x i32] poison, align 4
// CHECK-DAG: @{{.*}}device{{.*}} = internal addrspace(13) global [4 x i32] poison, align 4

// CHECK-LABEL: define {{.*}}@_Z6devicePii(
// CHECK: store i32 %{{.*}}, ptr addrspace(13) {{.*}}@{{.*}}device
__device__ void device(int *out, int i) {
  int dbuf[4] __attribute__((amdgpu_vgpr));
  dbuf[1] = i;
  out[0] = dbuf[1];
}

// CHECK-LABEL: define {{.*}}@_Z6kernelPii(
// CHECK: store i32 %{{.*}}, ptr addrspace(13) {{.*}}@{{.*}}kernel
// CHECK: load i32, ptr addrspace(13) {{.*}}@{{.*}}kernel
__global__ void kernel(int *out, int i) {
  int buf[4] __attribute__((amdgpu_vgpr));
  buf[2] = i;
  out[0] = buf[2];
  device(out, i);
}
