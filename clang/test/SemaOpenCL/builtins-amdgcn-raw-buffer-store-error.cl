// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char i8;
typedef short i16;
typedef int i32;
typedef int i64 __attribute__((ext_vector_type(2)));
typedef int i96 __attribute__((ext_vector_type(3)));
typedef int i128 __attribute__((ext_vector_type(4)));

void test_amdgcn_raw_ptr_buffer_store_b8(i8 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b8(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b8' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b16(i16 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b32(i32 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b64(i64 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b64(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b64' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b96(i96 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b96(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b96' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_b128(i128 vdata, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_buffer_store_b128(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_store_b128' must be a constant integer}}
}
