// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

float test_no_second_arg(float2 p0) {
  return smoothstep(p0);
  // expected-error@-1 {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 1 was provided}}
}

float test_no_third_arg(float2 p0) {
  return smoothstep(p0, p0);
  // expected-error@-1 {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 2 were provided}}
}

float test_too_many_arg(float2 p0) {
  return smoothstep(p0, p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 4 were provided}}
}

float test_double_inputs(double p0, double p1, double p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
}

float test_int_inputs(int p0, int p1, int p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored}}
}

float1 test_vec1_inputs(float1 p0, float1 p1, float1 p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 1>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 1>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, half>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 1]: no type named 'Type' in 'hlsl::__detail::enable_if<false, float>'}}
}

typedef float float5 __attribute__((ext_vector_type(5)));

float5 test_vec5_inputs(float5 p0, float5 p1, float5 p2) {
  return smoothstep(p0, p1, p2);
  // expected-error@-1  {{no matching function for call to 'smoothstep'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 5>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, vector<float, 5>>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, half>'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with N = 5]: no type named 'Type' in 'hlsl::__detail::enable_if<false, float>'}}
}
