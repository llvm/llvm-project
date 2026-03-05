#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:   -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:   -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// Test that AMDGPU kernel calling convention is correctly propagated
// from CIR through LLVM dialect to LLVM IR

// CIR: cir.func{{.*}}@kernel_simple(){{.*}}cc(amdgpu_kernel)
// LLVM: define{{.*}}amdgpu_kernel void @kernel_simple()
extern "C" __global__ void kernel_simple() {}

// CIR: cir.func{{.*}}@kernel_with_arg(%arg{{[0-9]+}}: !s32i{{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define{{.*}}amdgpu_kernel void @kernel_with_arg(i32{{.*}}%{{[0-9]+}})
extern "C" __global__ void kernel_with_arg(int x) {}

// Test device function
// CIR: cir.func{{.*}}@device_func(%arg{{[0-9]+}}: !s32i{{.*}})
// CIR-NOT: cc(amdgpu_kernel)
// LLVM: define{{.*}}void @device_func(i32{{.*}}%{{[0-9]+}})
extern "C" __device__ void device_func(int x) {}

// Test kernel with multiple arguments
// CIR: cir.func{{.*}}@kernel_multi_arg(%arg{{[0-9]+}}: !s32i{{.*}}, %arg{{[0-9]+}}: !cir.float{{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define{{.*}}amdgpu_kernel void @kernel_multi_arg(i32{{.*}}%{{[0-9]+}}, float{{.*}}%{{[0-9]+}})
extern "C" __global__ void kernel_multi_arg(int a, float b) {}

// Test that kernel can call device functions
// CIR: cir.func{{.*}}@kernel_calls_device(){{.*}}cc(amdgpu_kernel)
// LLVM: define{{.*}}amdgpu_kernel void @kernel_calls_device()
extern "C" __global__ void kernel_calls_device() {
  // CIR: cir.call @device_func{{.*}}(!s32i)
  // LLVM: call{{.*}}void @device_func(i32{{.*}}42)
  device_func(42);
}

