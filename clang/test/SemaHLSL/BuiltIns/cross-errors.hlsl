// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -disable-llvm-passes -verify

void test_too_few_arg()
{
  return cross();
  // expected-error@-1 {{no matching function for call to 'cross'}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 0 were provided}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 0 were provided}}
}

void test_too_few_arg_f32()
{
  return __builtin_hlsl_crossf32();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

void test_too_few_arg_f16()
{
  return __builtin_hlsl_crossf16();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

void test_too_many_arg(float3 p0)
{
  return cross(p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'cross'}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{candidate function not viable: requires 2 arguments, but 3 were provided}}
}

void test_too_many_arg_f32(float3 p0)
{
  return __builtin_hlsl_crossf32(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

void test_too_many_arg_f16(half3 p0)
{
  return __builtin_hlsl_crossf16(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

bool2 builtin_cross_int2_to_float2_promotion(int2 p1)
{
  return __builtin_hlsl_crossf32(p1, p1);
  // expected-error@-1 {{cannot initialize a parameter of type 'vector<float, 3>' (vector of 3 'float' values) with an lvalue of type 'int2' (aka 'vector<int, 2>')}}
}

float2 builtin_cross_float2(float2 p1, float2 p2)
{
  return __builtin_hlsl_crossf32(p1, p2);
  // expected-error@-1 {{cannot initialize a parameter of type 'vector<float, 3>' (vector of 3 'float' values) with an lvalue of type 'float2' (aka 'vector<float, 2>')}}
}

void test_ambiguous(int p0)
{
  return cross(p0,p0);
  // expected-error@-1 {{call to 'cross' is ambiguous}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_alias_intrinsics.h:* {{candidate function}}
}
