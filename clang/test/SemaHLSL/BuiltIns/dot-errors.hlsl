// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected=note

float test_no_second_arg(float2 p0) {
  return __builtin_hlsl_dot(p0);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

float test_too_many_arg(float2 p0) {
  return __builtin_hlsl_dot(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

float test_dot_no_second_arg(float2 p0) {
  return dot(p0);
  // expected-error@-1 {{no matching function for call to 'dot'}}
}

float test_dot_vector_size_mismatch(float3 p0, float2 p1) {
  return dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<[...], 3>' vs 'vector<[...], 2>')}}
}

float test_dot_builtin_vector_size_mismatch(float3 p0, float2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<[...], 3>' vs 'vector<[...], 2>')}}
}

float test_dot_scalar_mismatch(float p0, int p1) {
  return dot(p0, p1);
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

float test_dot_element_type_mismatch(int2 p0, float2 p1) {
  return dot(p0, p1);
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

//NOTE: for all the *_promotion we are intentionally not handling type promotion in builtins
float test_builtin_dot_vec_int_to_float_promotion(int2 p0, float2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<int, [...]>' vs 'vector<float, [...]>')}}
}

int64_t test_builtin_dot_vec_int_to_int64_promotion(int64_t2 p0, int2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<int64_t, [...]>' vs 'vector<int, [...]>')}}
}

float test_builtin_dot_vec_half_to_float_promotion(float2 p0, half2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<float, [...]>' vs 'vector<half, [...]>')}}
}

#ifdef __HLSL_ENABLE_16_BIT
float test_builtin_dot_vec_int16_to_float_promotion(float2 p0, int16_t2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<float, [...]>' vs 'vector<int16_t, [...]>')}}
}

half test_builtin_dot_vec_int16_to_half_promotion(half2 p0, int16_t2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<half, [...]>' vs 'vector<int16_t, [...]>')}}
}

int test_builtin_dot_vec_int16_to_int_promotion(int2 p0, int16_t2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<int, [...]>' vs 'vector<int16_t, [...]>')}}
}

int64_t test_builtin_dot_vec_int16_to_int64_promotion(int64_t2 p0,
                                                      int16_t2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('vector<int64_t, [...]>' vs 'vector<int16_t, [...]>')}}
}
#endif

float test_builtin_dot_float2_splat(float p0, float2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float' vs 'float2' (aka 'vector<float, 2>'))}}
}

float test_builtin_dot_float3_splat(float p0, float3 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float' vs 'float3' (aka 'vector<float, 3>'))}}
}

float test_builtin_dot_float4_splat(float p0, float4 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float' vs 'float4' (aka 'vector<float, 4>'))}}
}

float test_dot_float2_int_splat(float2 p0, int p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float2' (aka 'vector<float, 2>') vs 'int')}}
}

float test_dot_float3_int_splat(float3 p0, int p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('float3' (aka 'vector<float, 3>') vs 'int')}}
}

float test_builtin_dot_int_vect_to_float_vec_promotion(int2 p0, float p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{arguments are of different types ('int2' (aka 'vector<int, 2>') vs 'float')}}
}

int test_builtin_dot_bool_type_promotion(bool p0, float p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{1st argument must be a vector, integer or floating-point type (was 'bool')}}
}

double test_dot_double(double2 p0, double2 p1) {
  return dot(p0, p1);
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

double test_dot_double_builtin(double2 p0, double2 p1) {
  return __builtin_hlsl_dot(p0, p1);
  // expected-error@-1 {{1st argument must be a scalar floating-point type (was 'double2' (aka 'vector<double, 2>'))}}
}

float builtin_bool_to_float_type_promotion ( float p0, bool p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
   // expected-error@-1 {{are of different types ('float' vs 'bool')}}
}

float builtin_dot_int_to_float_promotion ( float p0, int p1 ) {
  return __builtin_hlsl_dot (p0, p1 );
  // expected-error@-1 {{are of different types ('float' vs 'int')}}
}
