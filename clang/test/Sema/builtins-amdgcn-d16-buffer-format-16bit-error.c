// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx700 -verify -fsyntax-only %s

// Verify that half typed buffer format load/store intrinsics require
// 16-bit-insts.

typedef _Float16 half;
typedef half half4 __attribute__((ext_vector_type(4)));

void test(half4 v, __amdgpu_buffer_rsrc_t rsrc) {
  v = __builtin_amdgcn_raw_buffer_load_format_v4f16( // expected-error {{needs target feature 16-bit-insts}}
      rsrc, 0, 0, 0);
  __builtin_amdgcn_raw_buffer_store_format_v4f16( // expected-error {{needs target feature 16-bit-insts}}
      v, rsrc, 0, 0, 0);
  v = __builtin_amdgcn_struct_buffer_load_format_v4f16( // expected-error {{needs target feature 16-bit-insts}}
      rsrc, 0, 0, 0, 0);
  __builtin_amdgcn_struct_buffer_store_format_v4f16( // expected-error {{needs target feature 16-bit-insts}}
      v, rsrc, 0, 0, 0, 0);
}
