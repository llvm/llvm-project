// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef _Float16 half;
typedef half half2 __attribute__((ext_vector_type(2)));

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

float test_float_and_half_inputs(float2 p0, half2 p1, float p2) {
  return __builtin_spirv_refract(p0, p1, p2);
  //  expected-error@-1 {{first two arguments to '__builtin_spirv_refract' must have the same type}}
}

float test_float_and_half_2_inputs(float2 p0, float2 p1, half p2) {
  return __builtin_spirv_refract(p0, p1, p2);
  //  expected-error@-1 {{all arguments to '__builtin_spirv_refract' must be of scalar or vector type with matching scalar element type: 'float2' (vector of 2 'float' values) vs 'half' (aka '_Float16')}}
}

float2 test_mismatch_vector_size_inputs(float2 p0, float3 p1, float p2) {
  return __builtin_spirv_refract(p0, p1, p2);
  //  expected-error@-1 {{first two arguments to '__builtin_spirv_refract' must have the same type}}
}
