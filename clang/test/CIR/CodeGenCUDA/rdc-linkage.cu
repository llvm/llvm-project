// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++17 -fgpu-rdc \
// RUN:   -cuid=abc -fclangir -emit-cir -x cuda %s -o - \
// RUN:   | FileCheck --check-prefix=CUDA-CIR %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++17 -fgpu-rdc \
// RUN:   -cuid=abc -fclangir -emit-llvm -x cuda %s -o - \
// RUN:   | FileCheck --check-prefix=CUDA-LLVM %s

// Host-side CUDA RDC registration is still handled by a later PR. Disable CIR
// passes here so this test only covers CIRGen's shadow linkage decisions.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple nvptx64-nvidia-cuda -std=c++17 -fgpu-rdc \
// RUN:   -cuid=abc -clangir-disable-passes -fclangir -emit-cir -x cuda %s -o - \
// RUN:   | FileCheck --check-prefix=CUDA-HOST-CIR %s

#include "Inputs/cuda.h"

extern "C" __device__ __host__ int use(int *);

// CUDA-HOST-CIR-DAG: cir.global external @device_var = #cir.poison
__device__ int device_var;

// CUDA-HOST-CIR-DAG: cir.global external @const_var = #cir.poison
__constant__ int const_var;

// CUDA-CIR-DAG: cir.global "private" external target_address_space(1) @_ZL17static_device_var__static__b04fd23c98500190
// CUDA-LLVM-DAG: @_ZL17static_device_var__static__b04fd23c98500190 = external addrspace(1) global i32
static __device__ int static_device_var;

// CUDA-CIR-DAG: cir.global "private" external target_address_space(4) @_ZL16static_const_var__static__b04fd23c98500190
// CUDA-LLVM-DAG: @_ZL16static_const_var__static__b04fd23c98500190 = external addrspace(4) global i32
static __constant__ int static_const_var;

namespace {
// CUDA-CIR-DAG: cir.func {{.*}} @_ZN12_GLOBAL__N_16kernelEv__intern__b04fd23c98500190()
// CUDA-LLVM-DAG: define weak_odr {{.*}}void @_ZN12_GLOBAL__N_16kernelEv__intern__b04fd23c98500190()
__global__ void kernel() {}
} // namespace

__device__ __host__ int touch() {
  return use(&static_device_var) + use((int *)&static_const_var);
}
