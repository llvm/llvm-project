// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -verify
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library %s -fnative-half-type -verify

float no_arg() {
  return __builtin_hlsl_elementwise_ddy_coarse();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float too_many_args(float val) {
  return __builtin_hlsl_elementwise_ddy_coarse(val, val);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float test_integer_scalar_input(int val) {
  return __builtin_hlsl_elementwise_ddy_coarse(val);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'int')}}
}

double test_double_scalar_input(double val) {
  return __builtin_hlsl_elementwise_ddy_coarse(val);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double')}}
}
