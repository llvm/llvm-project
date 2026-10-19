// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -disable-llvm-passes -verify

void test_too_few_arg()
{
  return step();
  // expected-error@-1 {{no matching function for call to 'step'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 2 arguments, but 0 were provided}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function template not viable: requires 2 arguments, but 0 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* 20 {{candidate function not viable: requires 2 arguments, but 0 were provided}}
}

void test_too_many_arg(float2 p0)
{
  return step(p0, p0, p0);
  // expected-error@-1 {{no matching function for call to 'step'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function template not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* 20 {{candidate function not viable: requires 2 arguments, but 3 were provided}}
}

bool test_bool_to_float_type_promotion(bool p1)
{
  return step(p1, p1);
  // expected-error@-1 {{call to 'step' is ambiguous}}
  // expected-note@hlsl/hlsl_compat_overloads.h:* 3 {{candidate function}}
}

float1 test_vec1_inputs(float1 p0, float1 p1)
{
  return step(p0, p1);
  // expected-warning@-1 2 {{implicit conversion turns vector to scalar: 'float1' (aka 'vector<float, 1>') to 'float'}}
}
