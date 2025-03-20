// RUN: %clang_cc1 %s -triple spirv-pc-vulkan-compute -verify

typedef float float2 __attribute__((ext_vector_type(2)));

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
