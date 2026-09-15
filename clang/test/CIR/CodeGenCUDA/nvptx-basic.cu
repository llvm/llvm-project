// Based on clang/test/CodeGenCUDA/ptx-kernels.cu tailored for CIR current capabilities.
// Tests basic device-side compilation with NVPTX target.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "Inputs/cuda.h"

// CHECK: cir.func {{.*}} @device_function()
extern "C"
__device__ void device_function() {}

// CHECK: cir.func {{.*}} @global_function()
// CHECK:   cir.call @device_function()
extern "C"
__global__ void global_function() {
  device_function();
}

// Template kernel with explicit instantiation
template <typename T> __global__ void templated_kernel(T param) {}
template __global__ void templated_kernel<int>(int);
// CHECK: cir.func {{.*}} @_Z16templated_kernelIiEvT_

// Anonymous namespace kernel
namespace {
__global__ void anonymous_ns_kernel() {}
// CHECK: cir.func {{.*}} @_ZN12_GLOBAL__N_119anonymous_ns_kernelEv
}
