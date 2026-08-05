// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned int v2u32 __attribute__((ext_vector_type(2)));
typedef unsigned int v3u32 __attribute__((ext_vector_type(3)));
typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));

void test_amdgcn_raw_ptr_buffer_store_b8(u8 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b8(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b8' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b16(u16 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b32(u32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b64(v2u32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b64(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b64' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b96(v3u32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b96(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b96' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b128(v4u32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b128(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b128' must be a constant integer}}
}
