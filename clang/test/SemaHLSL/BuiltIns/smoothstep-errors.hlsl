// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float test_no_second_arg(float2 p0) {
  return smoothstep(p0);
  // expected-error@-1 {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 3 arguments, but 1 was provided}}
}

float test_no_third_arg(float2 p0) {
  return smoothstep(p0, p0);
  // expected-error@-1 {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 3 arguments, but 2 were provided}}
}

float test_too_many_arg(float2 p0) {
  return smoothstep(p0, p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 3 arguments, but 4 were provided}}
}

float test_double_inputs(double p0, double p1, double p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{call to 'smoothstep' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float test_int_inputs(int p0, int p1, int p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{call to 'smoothstep' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float1 test_vec1_inputs(float1 p0, float1 p1, float1 p2) {
  return smoothstep(p0, p1, p2);
  // expected-warning@-1 3 {{implicit conversion turns vector to scalar: 'float1' (aka 'vector<float, 1>') to 'float'}}
}

typedef float float5 __attribute__((ext_vector_type(5)));

float5 test_vec5_inputs(float5 p0, float5 p1, float5 p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{call to 'smoothstep' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 4 {{candidate function}}
}
