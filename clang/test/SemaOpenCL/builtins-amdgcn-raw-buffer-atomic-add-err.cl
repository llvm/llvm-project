// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx90a -S -verify=gfx90a,expected -o - %s
// REQUIRES: amdgpu-registered-target

typedef half __attribute__((ext_vector_type(2))) float16x2_t;

void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, int i32, float f32, float16x2_t v2f16, int offset, int soffset, int x) {
  i32 = __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(i32, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32' must be a constant integer}}
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(f32, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32' must be a constant integer}}
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(v2f16, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16' must be a constant integer}}
}
