// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

bool test_too_few_arg() {
  return __builtin_hlsl_elementwise_sign();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

bool2 test_too_many_arg(float2 p0) {
  return __builtin_hlsl_elementwise_sign(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

bool builtin_sign_bool(bool p1) {
  return __builtin_hlsl_elementwise_sign(p1);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was 'bool')}}
}

int2 builtin_sign_bool2(bool2 p0) {
  return __builtin_hlsl_elementwise_sign(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer or floating-point types (was 'bool2' (aka 'vector<bool, 2>'))}}
}
