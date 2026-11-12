// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s -verify

// expected-note@*:* 108 {{candidate template ignored}}

// Test mul error cases via template overload resolution

// Inner dimension mismatch: vector * vector with different sizes
export float test_vec_dim_mismatch(float2 a, float3 b) {
  return mul(a, b);
  // expected-error@-1 {{no matching function for call to 'mul'}}
}

// Inner dimension mismatch: matrix * matrix
export float2x4 test_mat_dim_mismatch(float2x3 a, float4x4 b) {
  return mul(a, b);
  // expected-error@-1 {{no matching function for call to 'mul'}}
}

// Inner dimension mismatch: vector * matrix
export float3 test_vec_mat_mismatch(float3 v, float2x3 m) {
  return mul(v, m);
  // expected-error@-1 {{no matching function for call to 'mul'}}
}

// Inner dimension mismatch: matrix * vector
export float2 test_mat_vec_mismatch(float2x3 m, float2 v) {
  return mul(m, v);
  // expected-error@-1 {{no matching function for call to 'mul'}}
}

// Type mismatch: different element types
export float test_type_mismatch(float a, int b) {
  return mul(a, b);
  // expected-error@-1 {{no matching function for call to 'mul'}}
}

// Type mismatch: different vector element types
export float test_vec_type_mismatch(float3 a, int3 b) {
  return mul(a, b);
  // expected-error@-1 {{no matching function for call to 'mul'}}
}

