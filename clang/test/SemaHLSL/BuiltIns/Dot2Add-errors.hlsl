// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

float test_too_few_arg() {
  return __builtin_hlsl_dot2add();
  // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}
}

float test_too_many_arg(half2 p1, half2 p2, float p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3, p1);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

float test_float_arg2_type(half2 p1, float2 p2, float p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3);
  // expected-error@-1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(half)))) half' (vector of 2 'half' values)}}
}

float test_float_arg1_type(float2 p1, half2 p2, float p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3);
  // expected-error@-1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(half)))) half' (vector of 2 'half' values)}}
}

float test_double_arg3_type(half2 p1, half2 p2, double p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}

float test_float_arg1_arg2_type(float2 p1, float2 p2, float p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3);
  // expected-error@-1 {{passing 'float2' (aka 'vector<float, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(half)))) half' (vector of 2 'half' values)}}
}

float test_int16_arg1_arg2_type(int16_t2 p1, int16_t2 p2, float p3) {
  return __builtin_hlsl_dot2add(p1, p2, p3);
  // expected-error@-1 {{passing 'int16_t2' (aka 'vector<int16_t, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(half)))) half' (vector of 2 'half' values)}}
}
