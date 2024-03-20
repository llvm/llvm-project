// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1300 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

#define __global__ __attribute__((global))
#define __exp_amd_laneshared__ __attribute__((exp_amd_laneshared))

__exp_amd_laneshared__ float exchange[37];

// CHECK: @exchange = external addrspace(10) global [37 x float], align 4
__global__ void foo(float *a, float b) {
  b = b * exchange[7];
  *a = b;
}
