// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir \
// RUN:            -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:            -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ogcg.ll %s

//===----------------------------------------------------------------------===//
// Test ABI lowering from CIR to LLVM IR for AMDGPU OpenCL kernels
//===----------------------------------------------------------------------===//

// Test simple kernel
// CIR: cir.func{{.*}} @simple_kernel{{.*}} cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @simple_kernel()
// OGCG: define{{.*}} amdgpu_kernel void @simple_kernel()
__kernel void simple_kernel() {}

// Test kernel with int argument
// CIR: cir.func{{.*}} @kernel_with_int(%arg{{[0-9]+}}: !s32i{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @kernel_with_int(i32 %{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @kernel_with_int(i32{{.*}} %{{.*}})
__kernel void kernel_with_int(int x) {}

// Test kernel with pointer argument
// CIR: cir.func{{.*}} @kernel_with_ptr(%arg{{[0-9]+}}: !cir.ptr<!s32i, lang_address_space(offload_global)>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @kernel_with_ptr(ptr addrspace(1){{.*}}%{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @kernel_with_ptr(ptr addrspace(1){{.*}} %{{.*}})
__kernel void kernel_with_ptr(global int *ptr) {}

// Test kernel with multiple args
// CIR: cir.func{{.*}} @kernel_multi_arg(%arg{{[0-9]+}}: !s32i{{.*}}, %arg{{[0-9]+}}: !cir.float{{.*}}, %arg{{[0-9]+}}: !cir.ptr<!cir.float, lang_address_space(offload_global)>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @kernel_multi_arg(i32 %{{.*}}, float %{{.*}}, ptr addrspace(1){{.*}}%{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @kernel_multi_arg(i32{{.*}} %{{.*}}, float{{.*}} %{{.*}}, ptr addrspace(1){{.*}} %{{.*}})
__kernel void kernel_multi_arg(int a, float b, global float *c) {}

// Test device function
// CIR: cir.func{{.*}} @device_fn(%arg{{[0-9]+}}: !s32i{{.*}})
// CIR-NOT: cc(amdgpu_kernel)
// LLVM: define{{.*}} void @device_fn(i32 %{{.*}})
// LLVM-NOT: amdgpu_kernel
// OGCG: define{{.*}} void @device_fn(i32{{.*}} %{{.*}})
// OGCG-NOT: amdgpu_kernel
void device_fn(int x) {}

// Test device function with return value
// CIR: cir.func{{.*}} @device_fn_float(%arg{{[0-9]+}}: !cir.float{{.*}}) -> !cir.float
// LLVM: define{{.*}} float @device_fn_float(float %{{.*}})
// OGCG: define{{.*}} float @device_fn_float(float{{.*}} %{{.*}})
float device_fn_float(float f) { return f * 2.0f; }

// Test kernel with local address space pointer (addrspace 3)
// CIR: cir.func{{.*}} @kernel_local_ptr(%arg{{[0-9]+}}: !cir.ptr<!s32i, lang_address_space(offload_local)>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @kernel_local_ptr(ptr addrspace(3){{.*}}%{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @kernel_local_ptr(ptr addrspace(3){{.*}} %{{.*}})
__kernel void kernel_local_ptr(local int *ptr) {}

// Test kernel with constant address space pointer (addrspace 4)
// CIR: cir.func{{.*}} @kernel_constant_ptr(%arg{{[0-9]+}}: !cir.ptr<!s32i, lang_address_space(offload_constant)>{{.*}}) cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @kernel_constant_ptr(ptr addrspace(4){{.*}}%{{.*}})
// OGCG: define{{.*}} amdgpu_kernel void @kernel_constant_ptr(ptr addrspace(4){{.*}} %{{.*}})
__kernel void kernel_constant_ptr(constant int *ptr) {}
