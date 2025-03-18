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
  // expected-error@-1 {{1st argument must be a vector of 16 or 32 bit floating-point types (was 'bool')}}
}

bool builtin_cross_int_to_float_promotion(int p1)
{
  return __builtin_hlsl_cross(p1, p1);
  // expected-error@-1 {{1st argument must be a vector of 16 or 32 bit floating-point types (was 'int')}}
}

bool2 builtin_cross_int2_to_float2_promotion(int2 p1)
{
  return __builtin_hlsl_cross(p1, p1);
  // expected-error@-1 {{1st argument must be a vector of 16 or 32 bit floating-point types (was 'int2' (aka 'vector<int, 2>'))}}
}

float2 builtin_cross_float2(float2 p1, float2 p2)
{
  return __builtin_hlsl_cross(p1, p2);
  // expected-error@-1 {{too many elements in vector operand (expected 3 elements, have 2)}}
}

float3  builtin_cross_float3_int3(float3 p1, int3 p2)
{
  return __builtin_hlsl_cross(p1, p2);
  // expected-error@-1 {{2nd argument must be a vector of 16 or 32 bit floating-point types (was 'int3' (aka 'vector<int, 3>'))}}
}

half3 builtin_cross_same_type(half3 p0, float3 p1)
{
  return __builtin_hlsl_cross(p0, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_cross' must have the same type}}
}
