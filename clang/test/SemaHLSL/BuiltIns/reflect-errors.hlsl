// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

float test_no_second_arg(float2 p0) {
  return reflect(p0);
  // expected-error@-1 {{no matching function for call to 'reflect'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
}

float test_too_many_arg(float2 p0) {
  return reflect(p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'reflect'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires 2 arguments, but 3 were provided}}
}

float test_double_inputs(double p0, double p1) {
  return reflect(p0, p1);
  // expected-error@-1  {{call to 'reflect' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float test_int_inputs(int p0, int p1) {
  return reflect(p0, p1);
  // expected-error@-1  {{call to 'reflect' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}
