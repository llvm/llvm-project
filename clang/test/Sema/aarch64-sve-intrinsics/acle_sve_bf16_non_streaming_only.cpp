// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

__attribute__((target("bf16")))
void test_bf16(svfloat32_t svf32, svbfloat16_t svbf16)
{
  svbfmmla_f32(svf32, svbf16, svbf16);
}

void test_no_bf16(svfloat32_t svf32, svbfloat16_t svbf16)
{
  // expected-error@+1 {{'svbfmmla_f32' needs target feature sve,bf16}}
  svbfmmla_f32(svf32, svbf16, svbf16);
}

__attribute__((target("sme,bf16")))
void test_bf16_streaming(svfloat32_t svf32, svbfloat16_t svbf16) __arm_streaming
{
  // expected-error@+1 {{builtin can only be called from a non-streaming function}}
  svbfmmla_f32(svf32, svbf16, svbf16);
}
