// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility=default -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility=protected -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-PROTECTED %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility=hidden -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-HIDDEN %s

// Mirrors clang/test/CodeGenCUDA/amdgpu-visibility.cu for the SPIR-V AMDGCN
// target. Verifies that device kernels and variables with hidden visibility get
// upgraded to protected, matching native AMDGPU behavior.

#define __device__ __attribute__((device))
#define __constant__ __attribute__((constant))
#define __global__ __attribute__((global))

// CHECK-DEFAULT: @c ={{.*}} addrspace(1) externally_initialized constant
// CHECK-DEFAULT: @g ={{.*}} addrspace(1) externally_initialized global
// CHECK-PROTECTED: @c = protected addrspace(1) externally_initialized constant
// CHECK-PROTECTED: @g = protected addrspace(1) externally_initialized global
// CHECK-HIDDEN: @c = protected addrspace(1) externally_initialized constant
// CHECK-HIDDEN: @g = protected addrspace(1) externally_initialized global
__constant__ int c;
__device__ int g;

// CHECK-DEFAULT: @e = external addrspace(1) global
// CHECK-PROTECTED: @e = external protected addrspace(1) global
// CHECK-HIDDEN: @e = external protected addrspace(1) global
extern __device__ int e;

// dummy one to hold reference to `e`.
__device__ int f() {
  return e;
}

// CHECK-DEFAULT: define{{.*}} spir_kernel void @_Z3foov()
// CHECK-PROTECTED: define protected spir_kernel void @_Z3foov()
// CHECK-HIDDEN: define protected spir_kernel void @_Z3foov()
__global__ void foo() {
  g = c;
}
