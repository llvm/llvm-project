// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t.ogcg.ll

#include "Inputs/cuda.h"

// CIR: cir.func {{.*}} @device_function()
// LLVM: define{{.*}} void @device_function
// OGCG: define{{.*}} void @device_function
extern "C"
__device__ void device_function() {}

// CIR: cir.func {{.*}} @global_function() cc(ptx_kernel)
// LLVM: define{{.*}} ptx_kernel void @global_function
// OGCG: define{{.*}} ptx_kernel void @global_function
extern "C"
__global__ void global_function() {
  device_function();
}

template <typename T> __global__ void templated_kernel(T param) {}
template __global__ void templated_kernel<int>(int);
// CIR-DAG: cir.func {{.*}} @_Z16templated_kernelIiEvT_({{.*}}) cc(ptx_kernel)
// LLVM-DAG: define{{.*}} ptx_kernel void @_Z16templated_kernelIiEvT_(
// OGCG-DAG: define{{.*}} ptx_kernel void @_Z16templated_kernelIiEvT_(

namespace {
__global__ void anonymous_ns_kernel() {}
// CIR-DAG: cir.func {{.*}} @_ZN12_GLOBAL__N_119anonymous_ns_kernelEv() cc(ptx_kernel)
// LLVM-DAG: define{{.*}} ptx_kernel void @_ZN12_GLOBAL__N_119anonymous_ns_kernelEv(
// OGCG-DAG: define{{.*}} ptx_kernel void @_ZN12_GLOBAL__N_119anonymous_ns_kernelEv(
}
