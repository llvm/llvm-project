// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

void test_too_few_arg()
{
  return length();
  // expected-error@-1 {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires single argument 'X', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires single argument 'X', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but no arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but no arguments were provided}}
}

void test_too_many_arg(float2 p0)
{
  return length(p0, p0);
  // expected-error@-1 {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires single argument 'X', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires single argument 'X', but 2 arguments were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function template not viable: requires single argument 'X', but 2 arguments were provided}}
}

float double_to_float_type(double p0) {
  return length(p0);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}


float bool_to_float_type_promotion(bool p1)
{
  return length(p1);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float length_int_to_float_promotion(int p1)
{
  return length(p1);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float2 length_int2_to_float2_promotion(int2 p1)
{
  return length(p1);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}
