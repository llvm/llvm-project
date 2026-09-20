// RUN: %clang_cc1 -triple amdgpu7.00-unknown-unknown -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

// Half typed buffer format load/store builtins require d16 support
// (16-bit-insts), introduced in gfx8.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef half half4 __attribute__((ext_vector_type(4)));

void test(half4 v, __amdgpu_buffer_rsrc_t rsrc) {
  v = __builtin_amdgcn_raw_buffer_load_format_v4f16(rsrc, 0, 0, 0); // expected-error {{'__builtin_amdgcn_raw_buffer_load_format_v4f16' needs target feature 16-bit-insts}}
  __builtin_amdgcn_raw_buffer_store_format_v4f16(v, rsrc, 0, 0, 0); // expected-error {{'__builtin_amdgcn_raw_buffer_store_format_v4f16' needs target feature 16-bit-insts}}
  v = __builtin_amdgcn_struct_buffer_load_format_v4f16(rsrc, 0, 0, 0, 0); // expected-error {{'__builtin_amdgcn_struct_buffer_load_format_v4f16' needs target feature 16-bit-insts}}
  __builtin_amdgcn_struct_buffer_store_format_v4f16(v, rsrc, 0, 0, 0, 0); // expected-error {{'__builtin_amdgcn_struct_buffer_store_format_v4f16' needs target feature 16-bit-insts}}
}
