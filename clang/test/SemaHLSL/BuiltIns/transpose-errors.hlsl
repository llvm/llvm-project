// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s -verify

// Test transpose error cases

float test_transpose_scalar(float a) {
  return transpose(a);
  // expected-error@-1 {{no matching function for call to 'transpose'}}
}
// expected-note@*:* {{candidate template ignored}}
// expected-note@*:* {{candidate template ignored}}

float3 test_transpose_vector(float3 a) {
  return transpose(a);
  // expected-error@-1 {{no matching function for call to 'transpose'}}
}
// expected-note@*:* {{candidate template ignored}}
// expected-note@*:* {{candidate template ignored}}

void test_transpose_scalar_builtin(float a) {
  __builtin_hlsl_transpose(a);
  // expected-error@-1 {{1st argument must be a matrix type (was 'float')}}
}

void test_transpose_vector_builtin(float3 a) {
  __builtin_hlsl_transpose(a);
  // expected-error@-1 {{1st argument must be a matrix type (was 'float3' (aka 'vector<float, 3>'))}}
}

void test_transpose_int_builtin(int a) {
  __builtin_hlsl_transpose(a);
  // expected-error@-1 {{1st argument must be a matrix type (was 'int')}}
}
