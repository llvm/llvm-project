// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fhip-kernel-arg-name \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefix=NEG %s

#include "Inputs/cuda.h"

// CHECK: define{{.*}} amdgpu_kernel void @_Z6kerneliPf({{.*}} !kernel_arg_name [[MD:![0-9]+]]
// NEG-NOT: define{{.*}} amdgpu_kernel void @_Z6kerneliPf({{.*}} !kernel_arg_name
__global__ void kernel(int arg1, float *arg2) {
}

// CHECK: [[MD]] = !{!"arg1", !"arg2"}
