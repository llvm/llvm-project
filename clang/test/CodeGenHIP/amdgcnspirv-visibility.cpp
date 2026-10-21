// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility=default -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility=protected -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-PROTECTED %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fapply-global-visibility-to-externs -fvisibility=hidden -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-HIDDEN %s

// Mirrors clang/test/CodeGenCUDA/amdgpu-visibility.cu for the SPIR-V AMDGCN
// target. Verifies that device kernels and variables with hidden visibility get
// upgraded to protected, matching native AMDGPU behavior.

#define __device__ __attribute__((device))
#define __constant__ __attribute__((constant))
#define __global__ __attribute__((global))

// CHECK-DEFAULT-DAG: @c ={{.*}} addrspace(1) externally_initialized constant
// CHECK-DEFAULT-DAG: @g ={{.*}} addrspace(1) externally_initialized global
// CHECK-DEFAULT-DAG: @e = external addrspace(1) global
// CHECK-PROTECTED-DAG: @c = protected addrspace(1) externally_initialized constant
// CHECK-PROTECTED-DAG: @g = protected addrspace(1) externally_initialized global
// CHECK-PROTECTED-DAG: @e = external protected addrspace(1) global
// CHECK-HIDDEN-DAG: @c = protected addrspace(1) externally_initialized constant
// CHECK-HIDDEN-DAG: @g = protected addrspace(1) externally_initialized global
// CHECK-HIDDEN-DAG: @e = external protected addrspace(1) global
__constant__ int c;
__device__ int g;
extern __device__ int e;

// Explicit [[gnu::visibility("hidden")]] must be respected (not upgraded to
// protected), unlike the implicit -fvisibility=hidden flag.
// CHECK-DEFAULT-DAG: @h = hidden addrspace(1) externally_initialized global
// CHECK-PROTECTED-DAG: @h = hidden addrspace(1) externally_initialized global
// CHECK-HIDDEN-DAG: @h = hidden addrspace(1) externally_initialized global
__attribute__((visibility("hidden"))) __device__ int h;

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

// CHECK-DEFAULT: define hidden spir_kernel void @_Z3barv()
// CHECK-PROTECTED: define hidden spir_kernel void @_Z3barv()
// CHECK-HIDDEN: define hidden spir_kernel void @_Z3barv()
__attribute__((visibility("hidden"))) __global__ void bar() {
  h = 1;
}
