// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -disable-llvm-passes -verify -verify-ignore-unexpected

void test_too_few_arg()
{
  return __builtin_hlsl_check_access_fully_mapped();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

void test_too_many_arg(float2 p0)
{
  return __builtin_hlsl_check_access_fully_mapped(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 3}}
}

bool builtin_bool_to_float_type_promotion(bool p1)
{
  return __builtin_hlsl_check_access_fully_mapped(p1);
  // expected-error@-1 {{passing 'bool' to parameter of incompatible type 'unsigned int'}}
}


bool2 builtin_check_access_fully_mapped_int2_to_float2_promotion(int2 p1)
{
  return __builtin_hlsl_check_access_fully_mapped(p1);
  // expected-error@-1 {{passing 'int2' (aka 'vector<int, 2>') to parameter of incompatible type 'unsigned int'}}
}
