// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef float float2 __attribute__((ext_vector_type(2)));

void test_too_few_arg()
{
  return __builtin_spirv_fwidth();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float test_too_many_arg(float p0) {
  return __builtin_spirv_fwidth(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float test_int_scalar_inputs(int p0) {
  return __builtin_spirv_fwidth(p0);
  //  expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}
}

float test_mismatched_return(float2 p0) {
  return __builtin_spirv_fwidth(p0);
  // expected-error@-1 {{returning 'float2' (vector of 2 'float' values) from a function with incompatible result type 'float'}}
}
