// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned int v2u32 __attribute__((ext_vector_type(2)));
typedef unsigned int v3u32 __attribute__((ext_vector_type(3)));
typedef unsigned int v4u32 __attribute__((ext_vector_type(4)));

u8 test_amdgcn_raw_ptr_buffer_load_b8(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b8(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b8' must be a constant integer}}
}

u16 test_amdgcn_raw_ptr_buffer_load_b16(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b16(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b16' must be a constant integer}}
}

u32 test_amdgcn_raw_ptr_buffer_load_b32(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b32(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b32' must be a constant integer}}
}

v2u32 test_amdgcn_raw_ptr_buffer_load_b64(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b64(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b64' must be a constant integer}}
}

v3u32 test_amdgcn_raw_ptr_buffer_load_b96(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b96(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b96' must be a constant integer}}
}

v4u32 test_amdgcn_raw_ptr_buffer_load_b128(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, int aux) {
  return __builtin_amdgcn_raw_buffer_load_b128(rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_buffer_load_b128' must be a constant integer}}
}
