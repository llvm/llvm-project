// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef float float2 __attribute__((ext_vector_type(2)));

void test_too_few_arg()
{
  return __builtin_spirv_length();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

void test_too_many_arg(float2 p0)
{
  return __builtin_spirv_length(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float test_double_scalar_inputs(double p0) {
  return __builtin_spirv_length(p0);
  //  expected-error@-1 {{passing 'double' to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(double)))) double' (vector of 2 'double' values)}}
}

float test_int_scalar_inputs(int p0) {
  return __builtin_spirv_length(p0);
  //  expected-error@-1 {{passing 'int' to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(int)))) int' (vector of 2 'int' values)}}
}
