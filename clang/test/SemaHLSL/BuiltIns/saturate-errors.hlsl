// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected -Werror

float2 test_no_arg() {
  return saturate();
  // expected-error@-1 {{no matching function for call to 'saturate'}}
}

float2 test_too_many_arg(float2 p0) {
  return saturate(p0, p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'saturate'}}
}

float2 test_saturate_vector_size_mismatch(float3 p0) {
  return saturate(p0);
  // expected-error@-1 {{implicit conversion truncates vector: 'float3' (aka 'vector<float, 3>') to 'vector<float, 2>'}}
}

float2 test_saturate_float2_int_splat(int p0) {
  return saturate(p0);
  // expected-error@-1 {{call to 'saturate' is ambiguous}}
}

float2 test_saturate_int_vect_to_float_vec_promotion(int2 p0) {
  return saturate(p0);
  // expected-error@-1 {{call to 'saturate' is ambiguous}}
}

float test_saturate_bool_type_promotion(bool p0) {
  return saturate(p0);
  // expected-error@-1 {{call to 'saturate' is ambiguous}}
}
