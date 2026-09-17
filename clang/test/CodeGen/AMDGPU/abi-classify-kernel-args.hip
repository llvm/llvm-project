// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - %s | FileCheck %s --check-prefix=GCN
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefix=GCN
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - %s | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefix=SPIRV

// Kernel arguments are never byval, so aggregates go byref in the constant
// address space, which differs between amdgcn and SPIR-V.

#define __global__ __attribute__((global))

struct pair_t { int a, b; };
struct large_t { int a[64]; };
struct i32_wrapper_t { int i; };

__global__ void kern_scalar(int i, float f) {}
// GCN: define{{.*}} amdgpu_kernel void @_Z11kern_scalarif(i32 noundef %{{.*}}, float noundef %{{.*}})
// SPIRV: define{{.*}} spir_kernel void @_Z11kern_scalarif(i32 noundef %{{.*}}, float noundef %{{.*}})

__global__ void kern_ptr(int *p) {}
// GCN: define{{.*}} amdgpu_kernel void @_Z8kern_ptrPi(ptr addrspace(1) noundef %{{.*}})
// SPIRV: define{{.*}} spir_kernel void @_Z8kern_ptrPi(ptr addrspace(1) noundef %{{.*}})

__global__ void kern_atomic_ptr(_Atomic(int *) p) {}
// GCN: define{{.*}} amdgpu_kernel void @_Z15kern_atomic_ptrU7_AtomicPi(ptr addrspace(1) %{{.*}})
// SPIRV: define{{.*}} spir_kernel void @_Z15kern_atomic_ptrU7_AtomicPi(ptr addrspace(1) %{{.*}})

__global__ void kern_small(pair_t s) {}
// GCN: define{{.*}} amdgpu_kernel void @_Z10kern_small6pair_t(ptr addrspace(4) noundef byref(%struct.pair_t) align 4 %{{.*}})
// SPIRV: define{{.*}} spir_kernel void @_Z10kern_small6pair_t(ptr addrspace(2) noundef byref(%struct.pair_t) align 4 %{{.*}})

__global__ void kern_large(large_t s) {}
// GCN: define{{.*}} amdgpu_kernel void @_Z10kern_large7large_t(ptr addrspace(4) noundef byref(%struct.large_t) align 4 %{{.*}})
// SPIRV: define{{.*}} spir_kernel void @_Z10kern_large7large_t(ptr addrspace(2) noundef byref(%struct.large_t) align 4 %{{.*}})

__global__ void kern_wrapper(i32_wrapper_t s) {}
// GCN: define{{.*}} amdgpu_kernel void @_Z12kern_wrapper13i32_wrapper_t(i32 %{{.*}})
// SPIRV: define{{.*}} spir_kernel void @_Z12kern_wrapper13i32_wrapper_t(i32 %{{.*}})
