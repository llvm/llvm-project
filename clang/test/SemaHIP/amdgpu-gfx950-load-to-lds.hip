// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx950 -verify=device %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify=host %s
// device-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared)) 

__device__ void i_am_device(void* src, __amdgpu_buffer_rsrc_t rsrc, __shared__ void* dst, int vindex, int voffset, int soffset) {
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 1, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 2, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 4, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 12, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 16, voffset, soffset, 0, 0);

    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 1, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 2, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 4, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 12, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 16, vindex, voffset, soffset, 0, 0);

    __builtin_amdgcn_load_to_lds(src, dst, 1, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 2, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 4, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 12, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 16, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}

    __builtin_amdgcn_global_load_lds(src, dst, 1, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 2, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 4, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 12, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 16, 0 , 0);
}

__global__ void i_am_kernel(void* src, __amdgpu_buffer_rsrc_t rsrc, __shared__ void* dst, int vindex, int voffset, int soffset) {
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 1, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 2, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 4, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 12, voffset, soffset, 0, 0);
    __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 16, voffset, soffset, 0, 0);

    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 1, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 2, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 4, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 12, vindex, voffset, soffset, 0, 0);
    __builtin_amdgcn_struct_ptr_buffer_load_lds(rsrc, dst, 16, vindex, voffset, soffset, 0, 0);

    __builtin_amdgcn_load_to_lds(src, dst, 1, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 2, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 4, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 12, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}
    __builtin_amdgcn_load_to_lds(src, dst, 16, 0, 0); // host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) void *' with an lvalue of type 'void *'}}

    __builtin_amdgcn_global_load_lds(src, dst, 1, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 2, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 4, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 12, 0 , 0);
    __builtin_amdgcn_global_load_lds(src, dst, 16, 0 , 0);
}
