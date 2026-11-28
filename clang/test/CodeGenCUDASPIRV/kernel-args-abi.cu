// Checks how CUDA kernels on SPIR-V pass their arguments. The default address
// space is Generic.

// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -x cuda -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -x cuda -fexperimental-abi-lowering -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %if cir-enabled %{ %clang_cc1 -fcuda-is-device -triple spirv64 -x cuda -fclangir -emit-llvm %s -o - | FileCheck %s %}

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

struct Small { int a, b; };
struct NonCopyable {
  int i;
  __device__ NonCopyable(const NonCopyable &);
};

// Device function pointers keep the Generic address space.
// CHECK: define spir_func void @_Z7dev_ptrPi(ptr addrspace(4) noundef %{{[^,)]+}})
__device__ void dev_ptr(int *p) {}

// Generic kernel pointers are passed as CrossWorkgroup ones; the body still
// sees a Generic pointer.
// CHECK-LABEL: define spir_kernel void @_Z5k_ptrPi(ptr addrspace(1) noundef %{{[^,)]+}})
// CHECK:   call spir_func void @_Z7dev_ptrPi(ptr addrspace(4) noundef %{{[^,)]+}})
__global__ void k_ptr(int *p) { dev_ptr(p); }

// Pointers in any other address space are left alone.
// CHECK: define spir_kernel void @_Z12k_global_ptrPU3AS1i(ptr addrspace(1) noundef %{{[^,)]+}})
__global__ void k_global_ptr(__attribute__((address_space(1))) int *p) {}

// CHECK: define spir_kernel void @_Z13k_private_ptrPU3AS0i(ptr noundef %{{[^,)]+}})
__global__ void k_private_ptr(__attribute__((address_space(0))) int *p) {}

// Device compilations copy kernel records byval, even one that cannot be
// copied bitwise.
// CHECK: define spir_kernel void @_Z7k_small5Small(ptr noundef byval(%struct.Small) align 4 %{{[^,)]+}})
__global__ void k_small(Small s) {}

// CHECK: define spir_kernel void @_Z14k_non_copyable11NonCopyable(ptr noundef byval(%struct.NonCopyable) align 4 %{{[^,)]+}})
__global__ void k_non_copyable(NonCopyable n) {}

// Sub-word kernel arguments are extended.
// CHECK: define spir_kernel void @_Z5k_extcb(i8 noundef signext %{{[^,)]+}}, i1 noundef zeroext %{{[^,)]+}})
__global__ void k_ext(char c, bool b) {}
