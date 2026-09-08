// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -disable-llvm-passes -verify

void test_too_few_arg()
{
  return cross();
  // expected-error@-1 {{no matching function for call to 'cross'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function not viable: requires 2 arguments, but 0 were provided}}
}

void test_too_many_arg(float3 p0)
{
  return cross(p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'cross'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function not viable: requires 2 arguments, but 3 were provided}}
}

float2 test_cross_float2(float2 p1, float2 p2)
{
  return cross(p1, p2);
  // expected-error@-1 {{no matching function for call to 'cross'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

void test_ambiguous(int p0)
{
  return cross(p0,p0);
  // expected-error@-1 {{call to 'cross' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}
