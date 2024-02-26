// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -verify -verify-ignore-unexpected

float test_no_second_arg ( float2 p0) {
  return __builtin_hlsl_dot ( p0 );
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

float test_too_many_arg ( float2 p0) {
  return __builtin_hlsl_dot ( p0, p0, p0 );
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

float test_dot_no_second_arg ( float2 p0) {
  return dot ( p0 );
  // expected-error@-1 {{no matching function for call to 'dot'}}
}

float test_dot_vector_size_mismatch ( float3 p0, float2 p1 ) {
  return dot ( p0, p1 );
  // expected-warning@-1 {{implicit conversion truncates vector: 'float3' (aka 'vector<float, 3>') to 'float __attribute__((ext_vector_type(2)))' (vector of 2 'float' values)}}
}

float test_dot_builtin_vector_size_mismatch ( float3 p0, float2 p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

float test_dot_scalar_mismatch ( float p0, int p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

float test_dot_element_type_mismatch ( int2 p0, float2 p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

//NOTE: for all the *_promotion we are intentionally not handling type promotion in builtins
float test_builtin_dot_vec_int_to_float_promotion ( int2 p0, float2 p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

int64_t test_builtin_dot_vec_int_to_int64_promotion( int64_t2 p0, int2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

float test_builtin_dot_vec_half_to_float_promotion( float2 p0, half2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

#ifdef __HLSL_ENABLE_16_BIT
float test_builtin_dot_vec_int16_to_float_promotion( float2 p0, int16_t2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

half test_builtin_dot_vec_int16_to_half_promotion( half2 p0, int16_t2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

int test_builtin_dot_vec_int16_to_int_promotion( int2 p0, int16_t2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}

int64_t test_builtin_dot_vec_int16_to_int64_promotion( int64_t2 p0, int16_t2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must have the same type}}
}
#endif

float test_builtin_dot_float2_splat ( float p0, float2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must be vectors}}
}

float test_builtin_dot_float3_splat ( float p0, float3 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must be vectors}}
}

float test_builtin_dot_float4_splat ( float p0, float4 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must be vectors}}
}

float test_dot_float2_int_splat ( float2 p0, int p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must be vectors}}
}

float test_dot_float3_int_splat ( float3 p0, int p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must be vectors}}
}

float test_builtin_dot_int_vect_to_float_vec_promotion ( int2 p0, float p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to '__builtin_hlsl_dot' must be vectors}}
}

int test_builtin_dot_bool_type_promotion ( bool p0, bool p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'bool')}}
}
