// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

uint test_too_few_arg() {
  return __builtin_hlsl_wave_active_bit_or();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

uint test_too_many_arg(uint p0) {
  return __builtin_hlsl_wave_active_bit_or(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct Foo
{
  int a;
};

uint test_type_check(Foo p0) {
  return __builtin_hlsl_wave_active_bit_or(p0);
  // expected-error@-1 {{no viable conversion from 'Foo' to 'unsigned int'}}
}

float test_expr_bool_type_check(float p0) {
  return __builtin_hlsl_wave_active_bit_or(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'float')}}
}
