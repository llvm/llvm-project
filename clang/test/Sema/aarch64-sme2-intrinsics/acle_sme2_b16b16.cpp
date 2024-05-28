// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme2 -fsyntax-only -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s

#include <arm_sme.h>

void test_b16b16(svbool_t pg, uint64_t u64, int64_t i64, const bfloat16_t *const_bf16_ptr, bfloat16_t *bf16_ptr, svbfloat16_t bf16, svbfloat16x2_t bf16x2, svbfloat16x3_t bf16x3, svbfloat16x4_t bf16x4) __arm_streaming
{
  // expected-error@+1 {{'svclamp_single_bf16_x2' needs target feature sme2,b16b16}}
  svclamp_single_bf16_x2(bf16x2, bf16, bf16);
  // expected-error@+1 {{'svclamp_single_bf16_x4' needs target feature sme2,b16b16}}
  svclamp_single_bf16_x4(bf16x4, bf16, bf16);
}