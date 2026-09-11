// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float2x2 test_too_few_arg(float2x2 p0) {
  return __builtin_elementwise_min(p0);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

float2x2 test_too_many_arg(float2x2 p0) {
  return __builtin_elementwise_min(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

float2x2 test_mismatched_dims(float2x2 p0, float3x3 p1) {
  return __builtin_elementwise_min(p0, p1);
  // expected-error@-1 {{arguments are of different types ('matrix<[...], 2, 2>' vs 'matrix<[...], 3, 3>')}}
}

float2x2 test_mismatched_element_types(float2x2 p0, half2x2 p1) {
  return __builtin_elementwise_min(p0, p1);
  // expected-error@-1 {{arguments are of different types ('matrix<float, [2 * ...]>' vs 'matrix<half, [2 * ...]>')}}
}

float2x2 test_scalar_and_matrix(float p0, float2x2 p1) {
  return __builtin_elementwise_min(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float' vs 'float2x2' (aka 'matrix<float, 2, 2>'))}}
}

float2x2 test_vector_and_matrix(float2 p0, float2x2 p1) {
  return __builtin_elementwise_min(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float2' (aka 'vector<float, 2>') vs 'float2x2' (aka 'matrix<float, 2, 2>'))}}
}
