// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s -verify

// Test transpose error cases

export float test_transpose_scalar(float a) {
  return transpose(a);
  // expected-error@-1 {{no matching function for call to 'transpose'}}
}
// expected-note@*:* {{candidate template ignored}}
// expected-note@*:* {{candidate template ignored}}

export float3 test_transpose_vector(float3 a) {
  return transpose(a);
  // expected-error@-1 {{no matching function for call to 'transpose'}}
}
// expected-note@*:* {{candidate template ignored}}
// expected-note@*:* {{candidate template ignored}}

export void test_transpose_scalar_builtin(float a) {
  __builtin_hlsl_transpose(a);
  // expected-error@-1 {{1st argument must be a matrix type (was 'float')}}
}

export void test_transpose_vector_builtin(float3 a) {
  __builtin_hlsl_transpose(a);
  // expected-error@-1 {{1st argument must be a matrix type (was 'float3' (aka 'vector<float, 3>'))}}
}

export void test_transpose_int_builtin(int a) {
  __builtin_hlsl_transpose(a);
  // expected-error@-1 {{1st argument must be a matrix type (was 'int')}}
}
