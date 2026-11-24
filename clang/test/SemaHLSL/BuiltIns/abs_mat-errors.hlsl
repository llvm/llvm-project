// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float2x2 test_too_few_arg() {
  return __builtin_elementwise_abs();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float2x2 test_too_many_arg(float2x2 p0) {
  return __builtin_elementwise_abs(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float2x2 test_bool_matrix(bool2x2 p0) {
  return __builtin_elementwise_abs(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of signed integer or floating-point types (was 'bool2x2' (aka 'matrix<bool, 2, 2>'))}}
}

// The raw builtin rejects unsigned matrices.
float2x2 test_unsigned_matrix(uint2x2 p0) {
  return __builtin_elementwise_abs(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of signed integer or floating-point types (was 'uint2x2' (aka 'matrix<uint, 2, 2>'))}}
}
