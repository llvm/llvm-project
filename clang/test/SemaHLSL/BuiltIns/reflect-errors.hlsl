// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float test_no_second_arg(float2 p0) {
  return reflect(p0);
  // expected-error@-1 {{no matching function for call to 'reflect'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 2 arguments, but 1 was provided}}
}

float test_too_many_arg(float2 p0) {
  return reflect(p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'reflect'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 2 arguments, but 3 were provided}}
}

float test_double_inputs(double p0, double p1) {
  return reflect(p0, p1);
  // expected-error@-1  {{call to 'reflect' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float test_int_inputs(int p0, int p1) {
  return reflect(p0, p1);
  // expected-error@-1  {{call to 'reflect' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float1 test_vec1_inputs(float1 p0, float1 p1) {
  return reflect(p0, p1);
  // expected-warning@-1 2 {{implicit conversion turns vector to scalar: 'float1' (aka 'vector<float, 1>') to 'float'}}
}

typedef float float5 __attribute__((ext_vector_type(5)));

float5 test_vec5_inputs(float5 p0, float5 p1) {
  return reflect(p0, p1);
  // expected-error@-1  {{call to 'reflect' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 4 {{candidate function}}
}
