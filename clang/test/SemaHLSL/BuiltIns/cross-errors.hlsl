// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -disable-llvm-passes -verify

void test_too_few_arg()
{
  return __builtin_hlsl_cross();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

void test_too_many_arg(float3 p0)
{
  return __builtin_hlsl_cross(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

bool builtin_bool_to_float_type_promotion(bool p1)
{
  return __builtin_hlsl_cross(p1, p1);
  // expected-error@-1 {{passing 'bool' to parameter of incompatible type 'float'}}
}

bool builtin_cross_int_to_float_promotion(int p1)
{
  return __builtin_hlsl_cross(p1, p1);
  // expected-error@-1 {{passing 'int' to parameter of incompatible type 'float'}}
}

bool2 builtin_cross_int2_to_float2_promotion(int2 p1)
{
  return __builtin_hlsl_cross(p1, p1);
  // expected-error@-1 {{passing 'int2' (aka 'vector<int, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(float)))) float' (vector of 2 'float' values)}}
}

float2 builtin_cross_float2(float2 p1, float2 p2)
{
  return __builtin_hlsl_cross(p1, p2);
  // expected-error@-1 {{too many elements in vector operand (expected 3 elements, have 2)}}
}

float3  builtin_cross_float3_int3(float3 p1, int3 p2)
{
  return __builtin_hlsl_cross(p1, p2);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_cross' must have the same type}}
}
