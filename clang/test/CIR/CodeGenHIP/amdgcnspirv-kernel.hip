// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// Test that HIP kernels on AMDGCN-flavored SPIR-V get the spir_kernel
// calling convention.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

// CIR: cir.func{{.*}} @_Z13kernel_scalari{{.*}} cc(spir_kernel)
// LLVM: define spir_kernel void @_Z13kernel_scalari
__global__ void kernel_scalar(int a) {}

// CIR: cir.func{{.*}} @_Z9device_fni
// CIR-NOT: cc(spir_kernel)
// LLVM: define{{.*}} void @_Z9device_fni
__device__ void device_fn(int a) {}
