// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -verify -verify-ignore-unexpected

float2 test_no_second_arg(float2 p0) {
  return __builtin_hlsl_lerp(p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 1}}
}

float2 test_no_third_arg(float2 p0) {
  return __builtin_hlsl_lerp(p0, p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

float2 test_too_many_arg(float2 p0) {
  return __builtin_hlsl_lerp(p0, p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

float2 test_lerp_no_second_arg(float2 p0) {
  return lerp(p0);
  // expected-error@-1 {{no matching function for call to 'lerp'}}
}

float2 test_lerp_vector_size_mismatch(float3 p0, float2 p1) {
  return lerp(p0, p0, p1);
  // expected-warning@-1 {{implicit conversion truncates vector: 'float3' (aka 'vector<float, 3>') to 'float __attribute__((ext_vector_type(2)))' (vector of 2 'float' values)}}
}

float2 test_lerp_builtin_vector_size_mismatch(float3 p0, float2 p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must have vectors of the same type}}
}

float test_lerp_scalar_mismatch(float p0, half p1) {
  return lerp(p1, p0, p1);
  // expected-error@-1 {{call to 'lerp' is ambiguous}}
}

float2 test_lerp_element_type_mismatch(half2 p0, float2 p1) {
  return lerp(p1, p0, p1);
  // expected-error@-1 {{call to 'lerp' is ambiguous}}
}

float2 test_builtin_lerp_float2_splat(float p0, float2 p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must be vectors}}
}

float3 test_builtin_lerp_float3_splat(float p0, float3 p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must be vectors}}
}

float4 test_builtin_lerp_float4_splat(float p0, float4 p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must be vectors}}
}

float2 test_lerp_float2_int_splat(float2 p0, int p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must be vectors}}
}

float3 test_lerp_float3_int_splat(float3 p0, int p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must be vectors}}
}

float2 test_builtin_lerp_int_vect_to_float_vec_promotion(int2 p0, float p1) {
  return __builtin_hlsl_lerp(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_lerp' must be vectors}}
}

float test_builtin_lerp_bool_type_promotion(bool p0) {
  return __builtin_hlsl_lerp(p0, p0, p0);
  // expected-error@-1 {{1st argument must be a floating point type (was 'bool')}}
}

float builtin_bool_to_float_type_promotion(float p0, bool p1) {
  return __builtin_hlsl_lerp(p0, p0, p1);
  // expected-error@-1 {{3rd argument must be a floating point type (was 'bool')}}
}

float builtin_bool_to_float_type_promotion2(bool p0, float p1) {
  return __builtin_hlsl_lerp(p1, p0, p1);
  // expected-error@-1 {{2nd argument must be a floating point type (was 'bool')}}
}

float builtin_lerp_int_to_float_promotion(float p0, int p1) {
  return __builtin_hlsl_lerp(p0, p0, p1);
  // expected-error@-1 {{3rd argument must be a floating point type (was 'int')}}
}

float4 test_lerp_int4(int4 p0, int4 p1, int4 p2) {
  return __builtin_hlsl_lerp(p0, p1, p2);
   // expected-error@-1 {{1st argument must be a floating point type (was 'int4' (aka 'vector<int, 4>'))}}
}