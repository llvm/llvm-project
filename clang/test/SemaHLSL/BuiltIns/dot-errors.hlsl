// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -verify -verify-ignore-unexpected

float test_no_second_arg ( float2 p0) {
  return __builtin_hlsl_dot ( p0 );
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

float test_too_many_arg ( float2 p0) {
  return __builtin_hlsl_dot ( p0, p0, p0 );
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

//NOTE: eventually behavior should match builtin
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
  // expected-warning@-1 {{implicit conversion truncates vector: 'float3' (aka 'vector<float, 3>') to 'float2' (aka 'vector<float, 2>')}}
}


//NOTE: this case runs into the same problem as the below example
//int Fn1(int p0, int p1);
//int Fn1(float p0, float p1);
//int test_dot_scalar_mismatch ( float p0, int p1 ) {
//  return Fn1( p0, p1 );
//}
float test_dot_scalar_mismatch ( float p0, int p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

float test_dot_element_type_mismatch ( int2 p0, float2 p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}