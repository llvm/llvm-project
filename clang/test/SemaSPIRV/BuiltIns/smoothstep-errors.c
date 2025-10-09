// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

float2 test_no_second_arg(float2 p0) {
  return __builtin_spirv_smoothstep(p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 1}}
}

float2 test_no_third_arg(float2 p0) {
  return __builtin_spirv_smoothstep(p0, p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

float2 test_too_many_arg(float2 p0) {
  return __builtin_spirv_smoothstep(p0, p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

int test_int_scalar_inputs(int p0) {
  return __builtin_spirv_smoothstep(p0, p0, p0);
  //  expected-error@-1 {{1st argument must be a scalar or vector of floating-point types (was 'int')}}
}

float test_int_scalar_inputs2(float p0, int p1) {
  return __builtin_spirv_smoothstep(p0, p1, p1);
  //  expected-error@-1 {{all arguments to '__builtin_spirv_smoothstep' must have the same type}}
}

float test_int_scalar_inputs3(float p0, int p1) {
  return __builtin_spirv_smoothstep(p0, p0, p1);
  //  expected-error@-1 {{all arguments to '__builtin_spirv_smoothstep' must have the same type}}
}

float test_mismatched_arg(float p0, float2 p1) {
  return __builtin_spirv_smoothstep(p0, p1, p1);
  // expected-error@-1 {{all arguments to '__builtin_spirv_smoothstep' must have the same type}}
}

float test_mismatched_arg2(float p0, float2 p1) {
  return __builtin_spirv_smoothstep(p0, p0, p1);
  // expected-error@-1 {{all arguments to '__builtin_spirv_smoothstep' must have the same type}}
}

float test_mismatched_return(float2 p0) {
  return __builtin_spirv_smoothstep(p0, p0, p0);
  // expected-error@-1 {{returning 'float2' (vector of 2 'float' values) from a function with incompatible result type 'float'}}
}

float3 test_mismatched_return2(float2 p0) {
  return __builtin_spirv_smoothstep(p0, p0, p0);
  // expected-error@-1 {{returning 'float2' (vector of 2 'float' values) from a function with incompatible result type 'float3' (vector of 3 'float' values)}}
}
