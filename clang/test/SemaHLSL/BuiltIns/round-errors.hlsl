
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

float test_too_few_arg() {
  return __builtin_elementwise_round();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float2 test_too_many_arg(float2 p0) {
  return __builtin_elementwise_round(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float builtin_bool_to_float_type_promotion(bool p1) {
  return __builtin_elementwise_round(p1);
  // expected-error@-1 {{1st argument must be a floating point type (was 'bool')}}
}

float builtin_round_int_to_float_promotion(int p1) {
  return __builtin_elementwise_round(p1);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int')}}
}

float2 builtin_round_int2_to_float2_promotion(int2 p1) {
  return __builtin_elementwise_round(p1);
  // expected-error@-1 {{1st argument must be a floating point type (was 'int2' (aka 'vector<int, 2>'))}}
}
