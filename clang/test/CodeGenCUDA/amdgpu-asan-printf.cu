// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fsanitize=address \
// RUN:   -O3 -x hip | FileCheck -check-prefixes=MFCHECK %s

// MFCHECK: !{{.*}} = !{i32 4, !"amdgpu_hostcall", i32 1}

// Test to check hostcall module flag metadata is generated correctly
// when a program has printf call and compiled with -fsanitize=address.
#include "Inputs/cuda.h"
__device__ void non_kernel() {
  printf("sanitized device function");
}

__global__ void kernel() {
  non_kernel();
}

