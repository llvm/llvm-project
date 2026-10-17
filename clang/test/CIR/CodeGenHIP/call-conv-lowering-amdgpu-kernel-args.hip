// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies AMDGPU kernel arguments under the
// kernel convention.

// TODO(cir): Add the kernel argument cases still pending:
// - Multi-field aggregates, passed byref in addrspace(4), once CallConvLowering
//   supports IndirectAliased.
// - A multi-field struct holding a generic pointer. Classic codegen coerces
//   the pointer to global, the LLVM ABI library does not yet.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef struct { int i; } i32_wrapper_t;
typedef struct { i32_wrapper_t w; } nested_t;
typedef struct { float f[1]; } f32_array1_t;
typedef struct { float *x; } ptr_wrapper_t;
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef short short4 __attribute__((ext_vector_type(4)));

// Word-sized scalars pass through unchanged.
__global__ void kern_scalar(int i, float f) {}

// CIR: cir.func {{.*}}@_Z11kern_scalarif(%arg0: !s32i {{.*}}, %arg1: !cir.float {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_scalarif(i32 noundef %{{.*}}, float noundef %{{.*}})

__global__ void kern_wide(long l, double d) {}

// CIR: cir.func {{.*}}@_Z9kern_wideld(%arg0: !s64i {{.*}}, %arg1: !cir.double {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z9kern_wideld(i64 noundef %{{.*}}, double noundef %{{.*}})

__global__ void kern_half(_Float16 h) {}

// CIR: cir.func {{.*}}@_Z9kern_halfDF16_(%arg0: !cir.f16 {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z9kern_halfDF16_(half noundef %{{.*}})

__global__ void kern_bitint(_BitInt(65) b) {}

// CIR: cir.func {{.*}}@_Z11kern_bitintDB65_(%arg0: !cir.int<s, 65, bitint> {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_bitintDB65_(i65 noundef %{{.*}})

// Vectors, including 16-bit and 3-element ones, pass whole.
__global__ void kern_float4(float4 v) {}

// CIR: cir.func {{.*}}@_Z11kern_float4Dv4_f(%arg0: !cir.vector<4 x !cir.float> {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_float4Dv4_f(<4 x float> noundef %{{.*}})

__global__ void kern_short4(short4 v) {}

// CIR: cir.func {{.*}}@_Z11kern_short4Dv4_s(%arg0: !cir.vector<4 x !s16i> {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_short4Dv4_s(<4 x i16> noundef %{{.*}})

__global__ void kern_float3(float3 v) {}

// CIR: cir.func {{.*}}@_Z11kern_float3Dv3_f(%arg0: !cir.vector<3 x !cir.float> {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_float3Dv3_f(<3 x float> noundef %{{.*}})

// A pointer already in a non-generic address space keeps it.
__global__ void kern_global_ptr(__attribute__((address_space(1))) int *p) {}

// CIR: cir.func {{.*}}@_Z15kern_global_ptrPU3AS1i(%arg0: !cir.ptr<!s32i, target_address_space(1)> {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z15kern_global_ptrPU3AS1i(ptr addrspace(1) noundef %{{.*}})

// HIP passes a generic pointer or reference as a global pointer and casts it
// back to generic in the kernel.
__global__ void kern_ptr(int *p) {}

// CIR: cir.func {{.*}}@_Z8kern_ptrPi(%[[P:[^:]+]]: !cir.ptr<!s32i, target_address_space(1)> {{.*}}){{.*}}cc(amdgpu_kernel)
// CIR:   cir.cast address_space %[[P]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i>
// LLVM: define {{.*}}amdgpu_kernel void @_Z8kern_ptrPi(ptr addrspace(1) noundef %{{.*}})

__global__ void kern_ref(int &r) {}

// CIR: cir.func {{.*}}@_Z8kern_refRi(%{{.*}}: !cir.ptr<!s32i, target_address_space(1)> {{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z8kern_refRi(ptr addrspace(1) noundef nonnull align 4 dereferenceable(4) %{{.*}})

// A struct holding only a pointer is unwrapped first, so it is coerced too.
__global__ void kern_ptr_wrap(ptr_wrapper_t w) {}

// CIR: cir.func {{.*}}@_Z13kern_ptr_wrap13ptr_wrapper_t(%{{.*}}: !cir.ptr<{{.*}}, target_address_space(1)>{{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z13kern_ptr_wrap13ptr_wrapper_t(ptr addrspace(1) %{{.*}})

// A device function keeps the generic pointer.
__device__ void dev_ptr(int *p) {}

// CIR: cir.func {{.*}}@_Z7dev_ptrPi(%{{.*}}: !cir.ptr<!s32i> {{.*}})
// LLVM: define {{.*}}void @_Z7dev_ptrPi(ptr noundef %{{.*}})

// A single-element struct kernel argument is passed as its element.
__global__ void kern_wrapper(i32_wrapper_t w) {}

// CIR: cir.func {{.*}}@_Z12kern_wrapper13i32_wrapper_t(%arg0: !s32i{{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z12kern_wrapper13i32_wrapper_t(i32 %{{.*}})

// The element is found through nested structs and one-element arrays.
__global__ void kern_nested(nested_t n) {}

// CIR: cir.func {{.*}}@_Z11kern_nested8nested_t(%arg0: !s32i{{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_nested8nested_t(i32 %{{.*}})

__global__ void kern_array1(f32_array1_t a) {}

// CIR: cir.func {{.*}}@_Z11kern_array112f32_array1_t(%arg0: !cir.float{{.*}}){{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_array112f32_array1_t(float %{{.*}})

// Kernel arguments are read from the kernarg segment, so sub-word integers
// and bool are not extended, unlike in a device function.
__global__ void kern_char(char c) {}

// CIR: cir.func {{.*}}@_Z9kern_charc(%arg0: !s8i {llvm.noundef}{{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z9kern_charc(i8 noundef %{{.*}})

__global__ void kern_uchar(unsigned char c) {}

// CIR: cir.func {{.*}}@_Z10kern_ucharh(%arg0: !u8i {llvm.noundef}{{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z10kern_ucharh(i8 noundef %{{.*}})

__global__ void kern_short(short s) {}

// CIR: cir.func {{.*}}@_Z10kern_shorts(%arg0: !s16i {llvm.noundef}{{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z10kern_shorts(i16 noundef %{{.*}})

__global__ void kern_bool(bool b) {}

// CIR: cir.func {{.*}}@_Z9kern_boolb(%arg0: !cir.bool {llvm.noundef}{{.*}}cc(amdgpu_kernel)
// LLVM: define {{.*}}amdgpu_kernel void @_Z9kern_boolb(i1 noundef %{{.*}})

// The same argument to a device function is extended.
__device__ void dev_char(char c) {}

// CIR: cir.func {{.*}}@_Z8dev_charc(%arg0: !s8i {{.*}}llvm.signext
// LLVM: define {{.*}}void @_Z8dev_charc(i8 noundef signext %{{.*}})
