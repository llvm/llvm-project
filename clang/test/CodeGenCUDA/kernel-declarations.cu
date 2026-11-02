// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fgpu-rdc \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-cpu sm_89 \
// RUN:   -fcuda-is-device -fgpu-rdc -S -o - %s | FileCheck %s --check-prefix=PTX

#include "Inputs/cuda.h"

// Kernels declared in another translation unit must have the kernel calling
// convention even when only referenced by host-side launches.
extern "C" __global__ void kernel(int);
struct Arg {
  int x;
};
template <class T> __global__ void templated_kernel(T);
extern template __global__ void templated_kernel<Arg>(Arg);
struct Holder {
  static __global__ void member(int);
};

// IR-DAG: declare ptx_kernel void @kernel(i32 noundef)
// IR-DAG: declare ptx_kernel void @_Z16templated_kernelI3ArgEvT_(ptr noundef byval(%struct.Arg) align 4)
// IR-DAG: declare ptx_kernel void @_ZN6Holder6memberEi(i32 noundef)
// PTX-DAG: .extern .entry kernel
// PTX-DAG: .extern .entry _Z16templated_kernelI3ArgEvT_
// PTX-DAG: .extern .entry _ZN6Holder6memberEi

// Definitions continue to distinguish kernels from ordinary device functions.
// IR-DAG: define{{.*}} ptx_kernel void @defined_kernel()
// IR-DAG: define{{.*}} void @device_function()
// PTX-DAG: .visible .entry defined_kernel
// PTX-DAG: .visible .func device_function
extern "C" __global__ void defined_kernel() {}
extern "C" __device__ void device_function() {}

void launch() {
  kernel<<<1, 1>>>(1);
  templated_kernel<Arg><<<1, 1>>>(Arg{1});
  Holder::member<<<1, 1>>>(1);
  defined_kernel<<<1, 1>>>();
}
