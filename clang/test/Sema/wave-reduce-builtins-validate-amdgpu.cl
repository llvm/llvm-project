// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -fsyntax-only -verify %s

// Test that the second argument (strategy) must be a constant integer

void test_wave_reduce_u32(unsigned int val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add(val, 0);
  (void)__builtin_amdgcn_wave_reduce_sub(val, 1);
  (void)__builtin_amdgcn_wave_reduce_min(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor(val, 0);

  (void)__builtin_amdgcn_wave_reduce_add(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_sub(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_min(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_max(val, strategy); // expected-error {{expression is not an integer constant expression}}
}

void test_wave_reduce_i32(int val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_min(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor(val, 0);

  (void)__builtin_amdgcn_wave_reduce_min(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_max(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_and(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_or(val, strategy);  // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_xor(val, strategy); // expected-error {{expression is not an integer constant expression}}
}

void test_wave_reduce_u64(unsigned long val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add(val, 0);
  (void)__builtin_amdgcn_wave_reduce_sub(val, 1);
  (void)__builtin_amdgcn_wave_reduce_min(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor(val, 0);

  (void)__builtin_amdgcn_wave_reduce_add(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_sub(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_min(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_max(val, strategy); // expected-error {{expression is not an integer constant expression}}
}

void test_wave_reduce_i64(long val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_min(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max(val, 0);
  (void)__builtin_amdgcn_wave_reduce_and(val, 0);
  (void)__builtin_amdgcn_wave_reduce_or(val, 0);
  (void)__builtin_amdgcn_wave_reduce_xor(val, 0);

  (void)__builtin_amdgcn_wave_reduce_min(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_max(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_and(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_or(val, strategy);  // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_xor(val, strategy); // expected-error {{expression is not an integer constant expression}}
}

void test_wave_reduce_f32(float val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add(val, 0);
  (void)__builtin_amdgcn_wave_reduce_sub(val, 1);
  (void)__builtin_amdgcn_wave_reduce_min(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max(val, 0);

  (void)__builtin_amdgcn_wave_reduce_add(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_sub(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_min(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_max(val, strategy); // expected-error {{expression is not an integer constant expression}}
}

void test_wave_reduce_f64(double val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add(val, 0);
  (void)__builtin_amdgcn_wave_reduce_sub(val, 1);
  (void)__builtin_amdgcn_wave_reduce_min(val, 0);
  (void)__builtin_amdgcn_wave_reduce_max(val, 0);

  (void)__builtin_amdgcn_wave_reduce_add(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_sub(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_min(val, strategy); // expected-error {{expression is not an integer constant expression}}
  (void)__builtin_amdgcn_wave_reduce_max(val, strategy); // expected-error {{expression is not an integer constant expression}}
}

struct S { double x; long long y; };

void test_wave_reduce_struct(struct S val, int strategy) {
  (void)__builtin_amdgcn_wave_reduce_add(val, 0); // expected-error {{'val' argument must be a scalar integer or floating-point type (was '__private struct S')}}
  (void)__builtin_amdgcn_wave_reduce_sub(val, 1); // expected-error {{'val' argument must be a scalar integer or floating-point type (was '__private struct S')}}
  (void)__builtin_amdgcn_wave_reduce_min(val, 0); // expected-error {{'val' argument must be a scalar integer or floating-point type (was '__private struct S')}}
  (void)__builtin_amdgcn_wave_reduce_max(val, 0); // expected-error {{'val' argument must be a scalar integer or floating-point type (was '__private struct S')}}
}

void test_wave_reduce_invalid_strategy(int val) {
  (void)__builtin_amdgcn_wave_reduce_add(val, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 2]}}
  (void)__builtin_amdgcn_wave_reduce_sub(val, 3);  // expected-error {{argument value 3 is outside the valid range [0, 2]}}
}
