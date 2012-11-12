// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - | FileCheck %s

#include "../SemaCUDA/cuda.h"

// CHECK: define ptx_device{{.*}}device_function
__device__ void device_function() {}

// CHECK: define ptx_kernel{{.*}}global_function
__global__ void global_function() {
  // CHECK: call ptx_device{{.*}}device_function
  device_function();
}
