// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -verify=expected -o - %s
// REQUIRES: amdgpu-registered-target

void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, float f32, double f64, int offset, int soffset) {
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(f32, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32' needs target feature atomic-fmin-fmax-global-f32}}
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(f32, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32' needs target feature atomic-fmin-fmax-global-f32}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(f64, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64' needs target feature atomic-fmin-fmax-global-f64}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(f64, rsrc, offset, soffset, 0); // expected-error{{'__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64' needs target feature atomic-fmin-fmax-global-f64}}
}
