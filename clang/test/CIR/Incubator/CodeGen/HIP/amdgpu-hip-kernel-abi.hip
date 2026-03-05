#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir -fclangir-call-conv-lowering\
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir -fclangir-call-conv-lowering\
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

//===----------------------------------------------------------------------===//
// Test ABI lowering from CIR to LLVM IR for AMDGPU
//===----------------------------------------------------------------------===//

// Test simple kernel
// CIR: cir.func{{.*}} @_Z13simple_kernelv{{.*}} cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z13simple_kernelv()
// OGCG: define{{.*}} amdgpu_kernel void @_Z13simple_kernelv()
__global__ void simple_kernel() {}

// Test kernel with int argument
// CIR: cir.func{{.*}} @_Z14kernel_int_argi(%arg{{[0-9]+}}: !s32i{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z14kernel_int_argi(i32 %{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @_Z14kernel_int_argi(i32{{.*}} %{{.*}})
__global__ void kernel_int_arg(int x) {}

// Test kernel with char argument
// CIR: cir.func{{.*}} @_Z15kernel_char_argc(%arg{{[0-9]+}}: !s8i{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z15kernel_char_argc(i8 {{.*}}%{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @_Z15kernel_char_argc(i8{{.*}} %{{.*}})
__global__ void kernel_char_arg(char c) {}

// Test kernel with pointer (HIP coerces generic pointers to global addrspace 1)
// CIR: cir.func{{.*}} @_Z14kernel_ptr_argPi(%arg{{[0-9]+}}: !cir.ptr<!s32i{{.*}}>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z14kernel_ptr_argPi(ptr addrspace(1) %{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @_Z14kernel_ptr_argPi(ptr addrspace(1){{.*}} %{{.*}})
__global__ void kernel_ptr_arg(int *ptr) {}

// Test kernel with multiple args (pointer coerced to global addrspace 1)
// CIR: cir.func{{.*}} @_Z16kernel_multi_argifPf(%arg{{[0-9]+}}: !s32i{{.*}} !cir.float{{.*}} !cir.ptr<!cir.float{{.*}}>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z16kernel_multi_argifPf(i32 %{{.*}}, float %{{.*}}, ptr addrspace(1) %{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @_Z16kernel_multi_argifPf(i32{{.*}} %{{.*}}, float{{.*}} %{{.*}}, ptr addrspace(1){{.*}} %{{.*}})
__global__ void kernel_multi_arg(int a, float b, float *c) {}

// Test device function
// CIR: cir.func{{.*}} @_Z9device_fni(%arg{{[0-9]+}}: !s32i{{.*}})
// LLVM: define{{.*}} void @_Z9device_fni(i32 %{{.*}})
// OGCG: define{{.*}} void @_Z9device_fni(i32{{.*}} %{{.*}})
__device__ void device_fn(int x) {}

// Test device function with return value
// CIR: cir.func{{.*}} @_Z15device_fn_floatf(%arg{{[0-9]+}}: !cir.float{{.*}}) -> !cir.float
// LLVM: define{{.*}} float @_Z15device_fn_floatf(float %{{.*}})
// OGCG: define{{.*}} float @_Z15device_fn_floatf(float{{.*}} %{{.*}})
__device__ float device_fn_float(float f) { return f * 2.0f; }

// Test kernel with pointer (coerced to global addrspace 1)
// CIR: cir.func{{.*}} @_Z17kernel_shared_ptrPi(%arg{{[0-9]+}}: !cir.ptr<!s32i{{.*}}>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z17kernel_shared_ptrPi(ptr addrspace(1) %{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @_Z17kernel_shared_ptrPi(ptr addrspace(1){{.*}} %{{.*}})
__global__ void kernel_shared_ptr(int *ptr) {}

// Test variadic device function
// CIR: cir.func{{.*}} @_Z11variadic_fniz(%arg{{[0-9]+}}: !s32i{{.*}}, ...)
// LLVM: define{{.*}} void @_Z11variadic_fniz(i32 %{{.*}}, ...)
// OGCG: define{{.*}} void @_Z11variadic_fniz(i32{{.*}} %{{.*}}, ...)
__device__ void variadic_fn(int count, ...) {}
