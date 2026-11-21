// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies AMDGCN SPIR-V kernel arguments with
// the AMDGPU rules and the SPIR-V address spaces.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef struct { int i; } i32_wrapper_t;
typedef struct { int a, b; } pair_t;

__global__ void kern_scalar(int i, float f) {}

// CIR: cir.func {{.*}}@_Z11kern_scalarif(%arg0: !s32i {{.*}}, %arg1: !cir.float {{.*}}){{.*}}cc(spir_kernel)
// LLVM: define {{.*}}spir_kernel void @_Z11kern_scalarif(i32 noundef %{{.*}}, float noundef %{{.*}})

// Kernel arguments are not extended, unlike in a device function.
__global__ void kern_char(char c) {}

// CIR: cir.func {{.*}}@_Z9kern_charc(%arg0: !s8i {llvm.noundef}{{.*}}cc(spir_kernel)
// LLVM: define {{.*}}spir_kernel void @_Z9kern_charc(i8 noundef %{{.*}})

__device__ void dev_char(char c) {}

// CIR: cir.func {{.*}}@_Z8dev_charc(%arg0: !s8i {{.*}}llvm.signext
// LLVM: define {{.*}}spir_func void @_Z8dev_charc(i8 noundef signext %{{.*}})

// A generic pointer or reference is passed as a CrossWorkgroup pointer.
__global__ void kern_ptr(int *p) {}

// CIR: cir.func {{.*}}@_Z8kern_ptrPi(%{{.*}}: !cir.ptr<!s32i, target_address_space(1)> {{.*}}){{.*}}cc(spir_kernel)
// LLVM: define {{.*}}spir_kernel void @_Z8kern_ptrPi(ptr addrspace(1) noundef %{{.*}})

__global__ void kern_ref(int &r) {}

// CIR: cir.func {{.*}}@_Z8kern_refRi(%{{.*}}: !cir.ptr<!s32i, target_address_space(1)> {{.*}}){{.*}}cc(spir_kernel)
// LLVM: define {{.*}}spir_kernel void @_Z8kern_refRi(ptr addrspace(1) noundef align 4 dereferenceable(4) %{{.*}})

// A device function keeps the generic pointer.
__device__ void dev_ptr(int *p) {}

// CIR: cir.func {{.*}}@_Z7dev_ptrPi(%{{.*}}: !cir.ptr<!s32i, target_address_space(4)> {{.*}})
// LLVM: define {{.*}}spir_func void @_Z7dev_ptrPi(ptr addrspace(4) noundef %{{.*}})

// A single-element struct is passed as its element.
__global__ void kern_wrapper(i32_wrapper_t w) {}

// CIR: cir.func {{.*}}@_Z12kern_wrapper13i32_wrapper_t(%arg0: !s32i{{.*}}){{.*}}cc(spir_kernel)
// LLVM: define {{.*}}spir_kernel void @_Z12kern_wrapper13i32_wrapper_t(i32 %{{.*}})

// A multi-field struct is passed byref in UniformConstant.
__global__ void kern_pair(pair_t p) {}

// CIR: cir.func {{.*}}@_Z9kern_pair6pair_t(%{{.*}}: !cir.ptr<!rec_pair_t, target_address_space(2)> {llvm.align = 4 : i64, llvm.byref = !rec_pair_t, llvm.noundef}{{.*}}cc(spir_kernel)
// LLVM: define {{.*}}spir_kernel void @_Z9kern_pair6pair_t(ptr addrspace(2) noundef byref(%struct.pair_t) align 4 %{{.*}})
