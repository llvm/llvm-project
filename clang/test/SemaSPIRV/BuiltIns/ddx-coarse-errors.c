/// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef _Float16 half;
typedef float float2 __attribute__((ext_vector_type(2)));

float no_arg() {
  return __builtin_spirv_ddx_coarse();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

float too_many_args(float val) {
  return __builtin_spirv_ddx_coarse(val, val);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

float mismatched_return(float2 val) {
  return __builtin_spirv_ddx_coarse(val);
  // expected-error@-1 {{returning 'float2' (vector of 2 'float' values) from a function with incompatible result type 'float'}}
}

float test_integer_scalar_input(int val) {
  return __builtin_spirv_ddx_coarse(val);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'int')}}
}

double test_double_scalar_input(double val) {
  return __builtin_spirv_ddx_coarse(val);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double')}}
}