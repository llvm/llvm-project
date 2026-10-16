// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

void test_too_few_arg()
{
  return length();
  // expected-error@-1 {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires single argument 'X', but no arguments were provided}}
}

void test_too_many_arg(float2 p0)
{
  return length(p0, p0);
  // expected-error@-1 {{no matching function for call to 'length'}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 8 {{candidate function not viable: requires single argument 'X', but 2 arguments were provided}}
}

float double_to_float_type(double p0) {
  return length(p0);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}


float bool_to_float_type_promotion(bool p1)
{
  return length(p1);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float length_int_to_float_promotion(int p1)
{
  return length(p1);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float2 length_int2_to_float2_promotion(int2 p1)
{
  return length(p1);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 2 {{candidate function}}
}

float1 test_vec1_inputs(float1 p0) {
  return length(p0);
  // expected-warning@-1 {{implicit conversion turns vector to scalar: 'float1' (aka 'vector<float, 1>') to 'float'}}
}

typedef float float5 __attribute__((ext_vector_type(5)));

float5 test_vec5_inputs(float5 p0) {
  return length(p0);
  // expected-error@-1  {{call to 'length' is ambiguous}}
  // expected-note@hlsl/hlsl_inline_intrinsics_gen.inc:* 4 {{candidate function}}
}
