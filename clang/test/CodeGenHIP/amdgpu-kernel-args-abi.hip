// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -fexperimental-abi-lowering -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - | FileCheck %s %}

// Checks how HIP kernels take arguments through the kernarg segment.

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
// CHECK: %struct.ptr_pair_t = type { ptr, i32 }

typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef short short4 __attribute__((ext_vector_type(4)));

// Word-sized scalars pass through unchanged.
__global__ void kern_scalar(int i, float f) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_scalarif(i32 noundef %{{.*}}, float noundef %{{.*}})

__global__ void kern_wide(long l, double d) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z9kern_wideld(i64 noundef %{{.*}}, double noundef %{{.*}})

__global__ void kern_half(_Float16 h) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z9kern_halfDF16_(half noundef %{{.*}})

__global__ void kern_bitint(_BitInt(65) b) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_bitintDB65_(i65 noundef %{{.*}})

// Kernel arguments are read from the kernarg segment, so sub-word integers
// and bool are not extended, unlike in a device function.
__global__ void kern_char(char c) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z9kern_charc(i8 noundef %{{.*}})

__global__ void kern_uchar(unsigned char c) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z10kern_ucharh(i8 noundef %{{.*}})

__global__ void kern_short(short s) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z10kern_shorts(i16 noundef %{{.*}})

__global__ void kern_bool(bool b) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z9kern_boolb(i1 noundef %{{.*}})

// The same argument to a device function is extended.
__device__ void dev_char(char c) {}

// CHECK: define {{.*}}void @_Z8dev_charc(i8 noundef signext %{{.*}})

// Vectors, including 16-bit and 3-element ones, pass whole.
__global__ void kern_float4(float4 v) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_float4Dv4_f(<4 x float> noundef %{{.*}})

__global__ void kern_short4(short4 v) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_short4Dv4_s(<4 x i16> noundef %{{.*}})

__global__ void kern_float3(float3 v) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_float3Dv3_f(<3 x float> noundef %{{.*}})

// A pointer already in a non-generic address space keeps it.
__global__ void kern_global_ptr(__attribute__((address_space(1))) int *p) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z15kern_global_ptrPU3AS1i(ptr addrspace(1) noundef %{{.*}})

// HIP passes a generic pointer or reference as a global pointer and casts it
// back to generic in the kernel.
__global__ void kern_ptr(int *p) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z8kern_ptrPi(ptr addrspace(1) noundef %{{.*}})

__global__ void kern_ref(int &r) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z8kern_refRi(ptr addrspace(1) noundef nonnull align 4 dereferenceable(4) %{{.*}})

// A struct holding only a pointer is unwrapped first, so it is coerced too.
__global__ void kern_ptr_wrap(ptr_wrapper_t w) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z13kern_ptr_wrap13ptr_wrapper_t(ptr addrspace(1) %{{.*}})

// A device function keeps the generic pointer.
__device__ void dev_ptr(int *p) {}

// CHECK: define {{.*}}void @_Z7dev_ptrPi(ptr noundef %{{.*}})

// A single-element struct kernel argument is passed as its element.
__global__ void kern_wrapper(i32_wrapper_t w) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z12kern_wrapper13i32_wrapper_t(i32 %{{.*}})

// The element is found through nested structs and one-element arrays.
__global__ void kern_nested(nested_t n) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_nested8nested_t(i32 %{{.*}})

__global__ void kern_array1(f32_array1_t a) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_array112f32_array1_t(float %{{.*}})

// A multi-field struct is read in place from the kernarg segment, addrspace(4).
__global__ void kern_pair(pair_t p) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z9kern_pair6pair_t(ptr addrspace(4) noundef byref(%struct.pair_t) align 4 %[[PAIR:.*]])
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(4) align 4 %[[PAIR]], i64 8, i1 false)

// Kernels do not drop an empty struct. It is read from the kernarg segment.
__global__ void kern_empty(empty_t e) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z10kern_empty7empty_t(ptr addrspace(4) noundef byref(%struct.empty_t) align 1 %[[E:.*]])
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 1 %{{.*}}, ptr addrspace(4) align 1 %[[E]], i64 1, i1 false)

// A _Complex is an aggregate, so it is read in place too.
__global__ void kern_cfloat(_Complex float c) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z11kern_cfloatCf(ptr addrspace(4) noundef byref({ float, float }) align 4 %{{.*}})

// A pointer inside a multi-field struct stays generic.
__global__ void kern_ptr_pair(ptr_pair_t p) {}

// CHECK: define {{.*}}amdgpu_kernel void @_Z13kern_ptr_pair10ptr_pair_t(ptr addrspace(4) noundef byref(%struct.ptr_pair_t) align 8 %[[P:.*]])
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 8 %{{.*}}, ptr addrspace(4) align 8 %[[P]], i64 16, i1 false)
