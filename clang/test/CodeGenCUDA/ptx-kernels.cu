// Make sure that __global__ functions are emitted along with correct
// annotations and are added to @llvm.used to prevent their elimination.
// REQUIRES: nvptx-registered-target
//
// RUN: %clang_cc1 %s -triple nvptx-unknown-unknown -fcuda-is-device -emit-llvm -o - | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: define{{.*}} void @device_function
extern "C"
__device__ void device_function() {}

// CHECK-LABEL: define{{.*}} ptx_kernel void @global_function
extern "C"
__global__ void global_function() {
  // CHECK: call void @device_function
  device_function();
}

// Make sure host-instantiated kernels are preserved on device side.
template <typename T> __global__ void templated_kernel(T param) {}
// CHECK-DAG: define{{.*}} ptx_kernel void @_Z16templated_kernelIiEvT_(

namespace {
__global__ void anonymous_ns_kernel() {}
// CHECK-DAG: define{{.*}} void @_ZN12_GLOBAL__N_119anonymous_ns_kernelEv(
}

void host_function() {
  templated_kernel<<<0, 0>>>(0);
  anonymous_ns_kernel<<<0,0>>>();
}
