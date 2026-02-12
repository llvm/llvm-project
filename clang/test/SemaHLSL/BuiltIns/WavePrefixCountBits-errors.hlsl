// RUN: %clang_cc1 -finclude-default-header -fnative-int16-type -fnative-half-type \
// RUN:   -fmath-errno -ffp-contract=on -fno-rounding-math -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

int test_too_few_arg() {
  return __builtin_hlsl_wave_prefix_count_bits();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

int test_too_many_arg(bool p0) {
  return __builtin_hlsl_wave_prefix_count_bits(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float test_expr_bool_type_check(float p0) {
  return __builtin_hlsl_wave_prefix_count_bits(p0);
  // expected-error@-1 {{invalid operand of type 'float'}}
}

bool2 test_expr_bool_vec_type_check(bool2 p0) {
  return __builtin_hlsl_wave_prefix_count_bits(p0);
  // expected-error@-1 {{invalid operand of type 'bool2' (aka 'vector<bool, 2>')}}
}

struct S { float f; };

S test_expr_struct_type_check(S p0) {
  return __builtin_hlsl_wave_prefix_count_bits(p0);
  // expected-error@-1 {{invalid operand of type 'S'}}
}
