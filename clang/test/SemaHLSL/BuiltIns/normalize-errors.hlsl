// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -disable-llvm-passes -verify

void test_too_few_arg()
{
  return __builtin_hlsl_normalize();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

void test_too_many_arg(float2 p0)
{
  return __builtin_hlsl_normalize(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

bool builtin_bool_to_float_type_promotion(bool p1)
{
  return __builtin_hlsl_normalize(p1);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'bool')}}
}

bool builtin_normalize_int_to_float_promotion(int p1)
{
  return __builtin_hlsl_normalize(p1);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'int')}}
}

bool2 builtin_normalize_int2_to_float2_promotion(int2 p1)
{
  return __builtin_hlsl_normalize(p1);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'int2' (aka 'vector<int, 2>'))}}
}
