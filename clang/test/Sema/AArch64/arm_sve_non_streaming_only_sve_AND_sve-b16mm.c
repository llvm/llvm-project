// RUN: %clang_cc1 %s -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -target-feature +sve-b16mm -verify=guard

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

// Properties: guard="sve,sve-b16mm" streaming_guard="" flags=""

void test(void) {
  svbfloat16_t svbfloat16_t_val;

  svmmla(svbfloat16_t_val, svbfloat16_t_val, svbfloat16_t_val);
  svmmla_bf16(svbfloat16_t_val, svbfloat16_t_val, svbfloat16_t_val);
}

void test_streaming(void) __arm_streaming{
  svbfloat16_t svbfloat16_t_val;

  // guard-error@+1 {{builtin can only be called from a non-streaming function}}
  svmmla(svbfloat16_t_val, svbfloat16_t_val, svbfloat16_t_val);
  // guard-error@+1 {{builtin can only be called from a non-streaming function}}
  svmmla_bf16(svbfloat16_t_val, svbfloat16_t_val, svbfloat16_t_val);
}

void test_streaming_compatible(void) __arm_streaming_compatible{
  svbfloat16_t svbfloat16_t_val;

  // guard-error@+1 {{builtin can only be called from a non-streaming function}}
  svmmla(svbfloat16_t_val, svbfloat16_t_val, svbfloat16_t_val);
  // guard-error@+1 {{builtin can only be called from a non-streaming function}}
  svmmla_bf16(svbfloat16_t_val, svbfloat16_t_val, svbfloat16_t_val);
}
