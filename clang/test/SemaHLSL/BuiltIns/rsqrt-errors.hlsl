
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

float test_too_few_arg() {
  return __builtin_hlsl_elementwise_rsqrt();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float2 test_too_many_arg(float2 p0) {
  return __builtin_hlsl_elementwise_rsqrt(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float builtin_bool_to_float_type_promotion(bool p1) {
  return __builtin_hlsl_elementwise_rsqrt(p1);
  // expected-error@-1 {{passing 'bool' to parameter of incompatible type 'float'}}
}

float builtin_rsqrt_int_to_float_promotion(int p1) {
  return __builtin_hlsl_elementwise_rsqrt(p1);
  // expected-error@-1 {{passing 'int' to parameter of incompatible type 'float'}}
}

float2 builtin_rsqrt_int2_to_float2_promotion(int2 p1) {
  return __builtin_hlsl_elementwise_rsqrt(p1);
  // expected-error@-1 {{passing 'int2' (aka 'vector<int, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(float)))) float' (vector of 2 'float' values)}}
}

// builtins are variadic functions and so are subject to DefaultVariadicArgumentPromotion
half builtin_rsqrt_half_scalar (half p0) {
  return __builtin_hlsl_elementwise_rsqrt (p0);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}

float builtin_rsqrt_float_scalar ( float p0) {
  return __builtin_hlsl_elementwise_rsqrt (p0);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}
