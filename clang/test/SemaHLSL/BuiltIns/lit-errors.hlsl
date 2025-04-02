// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

float4 test_no_second_arg(float p0) {
  return lit(p0);
  // expected-error@-1 {{no matching function for call to 'lit'}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 1 was provided}}
}

float4 test_no_third_arg(float p0) {
  return lit(p0, p0);
  // expected-error@-1 {{no matching function for call to 'lit'}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 2 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 2 were provided}}
}

float4 test_too_many_arg(float p0) {
  return lit(p0, p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'lit'}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 4 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 3 arguments, but 4 were provided}}
}

float4 test_vec_inputs(float2 p0, float2 p1, float2 p2) {
  return lit(p0, p1, p2);
  // expected-error@-1  {{no matching function for call to 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float2]: invalid vector element type 'vector<float, 2>' (vector of 2 'float' values)}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float2]: invalid vector element type 'vector<float, 2>' (vector of 2 'float' values)}}
}

float4 test_vec1_inputs(float1 p0, float1 p1, float1 p2) {
  return lit(p0, p1, p2);
  // expected-error@-1  {{no matching function for call to 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float1]: invalid vector element type 'vector<float, 1>' (vector of 1 'float' value)}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate template ignored: substitution failure [with T = float1]: invalid vector element type 'vector<float, 1>' (vector of 1 'float' value)}}
}
