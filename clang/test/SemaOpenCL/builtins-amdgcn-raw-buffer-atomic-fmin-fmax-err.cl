// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx10-1-generic -S -verify=expected -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx10-3-generic -S -verify=expected -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1250 -S -verify=expected -o - %s

// REQUIRES: amdgpu-registered-target

void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, float f32, double f64, int offset, int soffset, int x) {
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(f32, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32' must be a constant integer}}
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(f32, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32' must be a constant integer}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(f64, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64' must be a constant integer}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(f64, rsrc, offset, soffset, x); // expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64' must be a constant integer}}
}
