// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -verify -verify-ignore-unexpected
// NOTE: This test is marked XFAIL because when overload resolution merges
// NOTE: test_dot_element_type_mismatch & test_dot_scalar_mismatch will have different behavior 
// XFAIL: *

float test_first_arg_is_not_vector ( float p0, float2 p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to 'dot' must be vectors}}
}

float test_second_arg_is_not_vector ( float2 p0, float p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to 'dot' must be vectors}}
}

int test_dot_unsupported_scalar_arg0 ( bool p0, int p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type (was 'bool')}}
}

int test_dot_unsupported_scalar_arg1 ( int p0, bool p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{2nd argument must be a vector, integer or floating point type (was 'bool')}}
}

float test_dot_scalar_mismatch ( float p0, int p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}

float test_dot_vector_size_mismatch ( float3 p0, float2 p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{first two arguments to 'dot' must have the same size}}
}

float test__no_second_arg ( float2 p0) {
  return dot ( p0 );
  // expected-error@-1 {{no matching function for call to 'dot'}}
}

float test_dot_element_type_mismatch ( int2 p0, float2 p1 ) {
  return dot ( p0, p1 );
  // expected-error@-1 {{call to 'dot' is ambiguous}}
}
