// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

bool2x2 test_too_few_arg() {
  return __builtin_hlsl_elementwise_isinf();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

bool2x2 test_too_many_arg(float2x2 p0) {
  return __builtin_hlsl_elementwise_isinf(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

bool2x2 test_builtin_isinf_int_matrix(int2x2 p0) {
  return __builtin_hlsl_elementwise_isinf(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'int2x2' (aka 'matrix<int, 2, 2>'))}}
}

bool2x2 test_builtin_isinf_double_matrix(double2x2 p0) {
  return __builtin_hlsl_elementwise_isinf(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double2x2' (aka 'matrix<double, 2, 2>'))}}
}
