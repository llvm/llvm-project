// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

bool test_too_few_arg() {
  return __builtin_hlsl_wave_read_lane_first();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float2 test_too_many_arg(float2 p0) {
  return __builtin_hlsl_wave_read_lane_first(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct S { float f; };

S test_expr_struct_type_check(S p0) {
  return __builtin_hlsl_wave_read_lane_first(p0);
  // expected-error@-1 {{invalid operand of type 'S' where a scalar, vector, or matrix is required}}
}

enum E { A };

E test_expr_enum_type_check(E p0) {
  return __builtin_hlsl_wave_read_lane_first(p0);
  // expected-error@-1 {{invalid operand of type 'E' where a scalar, vector, or matrix is required}}
}
