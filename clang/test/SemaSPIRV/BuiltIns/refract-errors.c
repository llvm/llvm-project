// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef float float2 __attribute__((ext_vector_type(2)));

float2 test_no_third_arg(float2 p0) {
  return __builtin_spirv_refract(p0, p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

float2 test_too_many_arg(float2 p0, float p1) {
  return __builtin_spirv_refract(p0, p0, p1, p1);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

float test_double_scalar_inputs(double p0, double p1, double p2) {
  return __builtin_spirv_refract(p0, p1, p2);
  //  expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double')}}
}

float test_int_scalar_inputs(int p0, int p1, int p2) {
  return __builtin_spirv_refract(p0, p1, p2);
  //  expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'int')}}
}
