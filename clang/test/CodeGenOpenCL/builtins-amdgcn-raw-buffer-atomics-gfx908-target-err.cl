// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx908 -S -verify=gfx908,expected -o - %s

// REQUIRES: amdgpu-registered-target

typedef half __attribute__((ext_vector_type(2))) float16x2_t;

void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, int i32, float f32, double f64, float16x2_t v2f16, int offset, int soffset) {
  i32 = __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(i32, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32' needs target feature gfx90a-insts}}
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(f32, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32' needs target feature gfx90a-insts}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f64(f64, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f64' needs target feature gfx90a-insts}}
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(v2f16, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16' needs target feature gfx90a-insts}}

  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(f32, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32' needs target feature gfx90a-insts}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(f64, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64' needs target feature gfx90a-insts}}
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_v2f16(v2f16, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_v2f16' needs target feature gfx90a-insts}}
}
