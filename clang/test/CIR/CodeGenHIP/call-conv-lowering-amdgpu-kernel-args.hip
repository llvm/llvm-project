// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies AMDGPU kernel arguments under the
// kernel convention.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef struct { int i; } i32_wrapper_t;
typedef struct { i32_wrapper_t w; } nested_t;
typedef struct { float f[1]; } f32_array1_t;
typedef struct { float *x; } ptr_wrapper_t;
typedef struct { int a; int b; } pair_t;
typedef struct {} empty_t;
typedef struct { float *x; int n; } ptr_pair_t;

// Unlike ptr_wrapper_t, the pointer field is not coerced to addrspace(1).
// CIR: !rec_ptr_pair_t = !cir.struct<"ptr_pair_t" {data !cir.ptr<!cir.float>, data !s32i}>
// LLVM: %struct.ptr_pair_t = type { ptr, i32 }

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

// A multi-field struct is read in place from the kernarg segment, addrspace(4).
__global__ void kern_pair(pair_t p) {}

// CIR: cir.func {{.*}}@_Z9kern_pair6pair_t(%[[PAIR:[^:]+]]: !cir.ptr<!rec_pair_t, target_address_space(4)> {{.*}}llvm.byref = !rec_pair_t{{.*}}cc(amdgpu_kernel)
// CIR:   cir.copy %[[PAIR]] align(4) to %{{.*}} align(4) : !cir.ptr<!rec_pair_t, target_address_space(4)>, !cir.ptr<!rec_pair_t>
// LLVM: define {{.*}}amdgpu_kernel void @_Z9kern_pair6pair_t(ptr addrspace(4) noundef byref(%struct.pair_t) align 4 %[[PAIR:.*]])
// LLVM:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(4) align 4 %[[PAIR]], i64 8, i1 false)

// Kernels do not drop an empty struct. It is read from the kernarg segment.
__global__ void kern_empty(empty_t e) {}

// CIR: cir.func {{.*}}@_Z10kern_empty7empty_t(%[[E:[^:]+]]: !cir.ptr<!rec_empty_t, target_address_space(4)> {{.*}}llvm.byref = !rec_empty_t{{.*}}cc(amdgpu_kernel)
// CIR:   cir.copy %[[E]] align(1) to %{{.*}} align(1) : !cir.ptr<!rec_empty_t, target_address_space(4)>, !cir.ptr<!rec_empty_t>
// LLVM: define {{.*}}amdgpu_kernel void @_Z10kern_empty7empty_t(ptr addrspace(4) noundef byref(%struct.empty_t) align 1 %[[E:.*]])
// LLVM:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 1 %{{.*}}, ptr addrspace(4) align 1 %[[E]], i64 1, i1 false)

// A _Complex is an aggregate, so it is read in place too.
__global__ void kern_cfloat(_Complex float c) {}

// CIR: cir.func {{.*}}@_Z11kern_cfloatCf(%[[C:[^:]+]]: !cir.ptr<!cir.complex<!cir.float>, target_address_space(4)> {{.*}}llvm.byref = !cir.complex<!cir.float>{{.*}}cc(amdgpu_kernel)
// CIR-NEXT:   cir.load %[[C]] : !cir.ptr<!cir.complex<!cir.float>, target_address_space(4)>, !cir.complex<!cir.float>
// LLVM: define {{.*}}amdgpu_kernel void @_Z11kern_cfloatCf(ptr addrspace(4) noundef byref({ float, float }) align 4 %{{.*}})

// A pointer inside a multi-field struct stays generic.
__global__ void kern_ptr_pair(ptr_pair_t p) {}

// CIR: cir.func {{.*}}@_Z13kern_ptr_pair10ptr_pair_t(%[[P:[^:]+]]: !cir.ptr<!rec_ptr_pair_t, target_address_space(4)> {{.*}}llvm.byref = !rec_ptr_pair_t{{.*}}cc(amdgpu_kernel)
// CIR:   cir.copy %[[P]] align(8) to %{{.*}} align(8) : !cir.ptr<!rec_ptr_pair_t, target_address_space(4)>, !cir.ptr<!rec_ptr_pair_t>
// LLVM: define {{.*}}amdgpu_kernel void @_Z13kern_ptr_pair10ptr_pair_t(ptr addrspace(4) noundef byref(%struct.ptr_pair_t) align 8 %[[P:.*]])
// LLVM:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 8 %{{.*}}, ptr addrspace(4) align 8 %[[P]], i64 16, i1 false)

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
