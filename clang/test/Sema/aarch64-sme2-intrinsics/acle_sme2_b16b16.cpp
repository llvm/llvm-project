// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +bf16 -target-feature +sme -target-feature +sme2 -emit-llvm-only -verify -verify-ignore-unexpected=error,note -o - %s

#include <arm_sme.h>

void test_b16b16( svbfloat16_t bf16, svbfloat16x2_t bf16x2, svbfloat16x4_t bf16x4) __arm_streaming
{
  // expected-error@+1 {{'svclamp_single_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svclamp_single_bf16_x2(bf16x2, bf16, bf16);
  // expected-error@+1 {{'svclamp_single_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svclamp_single_bf16_x4(bf16x4, bf16, bf16);

  // expected-error@+1 {{'svmax_single_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svmax_single_bf16_x2(bf16x2, bf16);
  // expected-error@+1 {{'svmax_single_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svmax_single_bf16_x4(bf16x4, bf16);
  // expected-error@+1 {{'svmax_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svmax_bf16_x2(bf16x2, bf16x2);
  // expected-error@+1 {{'svmax_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svmax_bf16_x4(bf16x4, bf16x4);

  // expected-error@+1 {{'svmaxnm_single_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svmaxnm_single_bf16_x2(bf16x2, bf16);
  // expected-error@+1 {{'svmaxnm_single_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svmaxnm_single_bf16_x4(bf16x4, bf16);
  // expected-error@+1 {{'svmaxnm_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svmaxnm_bf16_x2(bf16x2, bf16x2);
  // expected-error@+1 {{'svmaxnm_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svmaxnm_bf16_x4(bf16x4, bf16x4);

  // expected-error@+1 {{'svmin_single_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svmin_single_bf16_x2(bf16x2, bf16);
  // expected-error@+1 {{'svmin_single_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svmin_single_bf16_x4(bf16x4, bf16);
  // expected-error@+1 {{'svmin_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svmin_bf16_x2(bf16x2, bf16x2);
  // expected-error@+1 {{'svmin_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svmin_bf16_x4(bf16x4, bf16x4);

  // expected-error@+1 {{'svminnm_single_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svminnm_single_bf16_x2(bf16x2, bf16);
  // expected-error@+1 {{'svminnm_single_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svminnm_single_bf16_x4(bf16x4, bf16);

  // expected-error@+1 {{'svminnm_bf16_x2' needs target feature sme,sme2,sve-b16b16}}
  svminnm_bf16_x2(bf16x2, bf16x2);
  // expected-error@+1 {{'svminnm_bf16_x4' needs target feature sme,sme2,sve-b16b16}}
  svminnm_bf16_x4(bf16x4, bf16x4);
}
