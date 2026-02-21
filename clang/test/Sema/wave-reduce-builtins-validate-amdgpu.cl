// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -fsyntax-only -verify %s

// Test that the second argument (strategy) must be a constant integer

void test_wave_reduce_u32(unsigned int val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add_u32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_sub_u32(val, 1);
  (void)__builtin_amdgcn_wave_reduce_min_u32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max_u32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and_b32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or_b32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor_b32(val, 0);
  
  (void)__builtin_amdgcn_wave_reduce_add_u32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_add_u32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_sub_u32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_sub_u32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_min_u32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_min_u32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_max_u32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_max_u32' must be a constant integer}}
}

void test_wave_reduce_i32(int val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_min_i32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max_i32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and_b32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or_b32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor_b32(val, 0);
  
  (void)__builtin_amdgcn_wave_reduce_min_i32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_min_i32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_max_i32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_max_i32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_and_b32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_and_b32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_or_b32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_or_b32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_xor_b32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_xor_b32' must be a constant integer}}
}

void test_wave_reduce_u64(unsigned long val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add_u64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_sub_u64(val, 1);
  (void)__builtin_amdgcn_wave_reduce_min_u64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max_u64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and_b64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or_b64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor_b64(val, 0);
  
  (void)__builtin_amdgcn_wave_reduce_add_u64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_add_u64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_sub_u64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_sub_u64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_min_u64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_min_u64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_max_u64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_max_u64' must be a constant integer}}
}

void test_wave_reduce_i64(long val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_min_i64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max_i64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and_b64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or_b64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor_b64(val, 0);
  
  (void)__builtin_amdgcn_wave_reduce_min_i64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_min_i64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_max_i64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_max_i64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_and_b64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_and_b64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_or_b64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_or_b64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_xor_b64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_xor_b64' must be a constant integer}}
}

void test_wave_reduce_f32(float val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_fadd_f32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_fsub_f32(val, 1);
  (void)__builtin_amdgcn_wave_reduce_fmin_f32(val, 0);
  (void)__builtin_amdgcn_wave_reduce_fmax_f32(val, 0);
  
  (void)__builtin_amdgcn_wave_reduce_fadd_f32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fadd_f32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_fsub_f32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fsub_f32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_fmin_f32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fmin_f32' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_fmax_f32(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fmax_f32' must be a constant integer}}
}

void test_wave_reduce_f64(double val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_fadd_f64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_fsub_f64(val, 1);
  (void)__builtin_amdgcn_wave_reduce_fmin_f64(val, 0);
  (void)__builtin_amdgcn_wave_reduce_fmax_f64(val, 0);
  
  (void)__builtin_amdgcn_wave_reduce_fadd_f64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fadd_f64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_fsub_f64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fsub_f64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_fmin_f64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fmin_f64' must be a constant integer}}
  (void)__builtin_amdgcn_wave_reduce_fmax_f64(val, strategy); // expected-error {{argument to '__builtin_amdgcn_wave_reduce_fmax_f64' must be a constant integer}}
}
