// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

uint test_too_few_arg() {
  return __builtin_hlsl_wave_active_bit_or();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

uint test_too_many_arg(uint p0) {
  return __builtin_hlsl_wave_active_bit_or(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct S { uint x; };

uint test_expr_struct_type_check(S p0) {
  return __builtin_hlsl_wave_active_bit_or(p0);
  // expected-error@-1 {{invalid operand of type 'S' where a scalar or vector is required}}
}

bool test_expr_bool_type_check(bool p0) {
  return __builtin_hlsl_wave_active_bit_or(p0);
  // expected-error@-1 {{invalid operand of type 'bool'}}
}
