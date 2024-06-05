// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -verify -emit-llvm-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>


void test_features_f16f16(uint32_t slice,
                          svfloat16_t zm,
                          svfloat16x2_t zn2, svfloat16x2_t zm2,
                          svfloat16x4_t zn4, svfloat16x4_t zm4,
                          svbfloat16_t bzm,
                          svbfloat16x2_t bzn2, svbfloat16x2_t bzm2,
                          svbfloat16x4_t bzn4, svbfloat16x4_t bzm4)

   __arm_streaming __arm_inout("za") {
  // expected-error@+1 {{'svmla_single_za16_f16_vg1x2' needs target feature sme-f16f16}}
  svmla_single_za16_f16_vg1x2(slice, zn2, zm);
  // expected-error@+1 {{'svmla_single_za16_f16_vg1x4' needs target feature sme-f16f16}}
  svmla_single_za16_f16_vg1x4(slice, zn4, zm);
  // expected-error@+1 {{'svmls_single_za16_f16_vg1x2' needs target feature sme-f16f16}}
  svmls_single_za16_f16_vg1x2(slice, zn2, zm);
  // expected-error@+1 {{'svmls_single_za16_f16_vg1x4' needs target feature sme-f16f16}}
  svmls_single_za16_f16_vg1x4(slice, zn4, zm);
  // expected-error@+1 {{'svmla_za16_f16_vg1x2' needs target feature sme-f16f16}}
  svmla_za16_f16_vg1x2(slice, zn2, zm2);
  // expected-error@+1 {{'svmla_za16_f16_vg1x4' needs target feature sme-f16f16}}
  svmla_za16_f16_vg1x4(slice, zn4, zm4);
  // expected-error@+1 {{'svmls_za16_f16_vg1x2' needs target feature sme-f16f16}}
  svmls_za16_f16_vg1x2(slice, zn2, zm2);
  // expected-error@+1 {{'svmls_za16_f16_vg1x4' needs target feature sme-f16f16}}
  svmls_za16_f16_vg1x4(slice, zn4, zm4);
  // expected-error@+1 {{'svmla_lane_za16_f16_vg1x2' needs target feature sme-f16f16}}
  svmla_lane_za16_f16_vg1x2(slice, zn2, zm, 7);
  // expected-error@+1 {{'svmla_lane_za16_f16_vg1x4' needs target feature sme-f16f16}}
  svmla_lane_za16_f16_vg1x4(slice, zn4, zm, 7);
  // expected-error@+1 {{'svmls_lane_za16_f16_vg1x2' needs target feature sme-f16f16}}
  svmls_lane_za16_f16_vg1x2(slice, zn2, zm, 7);
  // expected-error@+1 {{'svmls_lane_za16_f16_vg1x4' needs target feature sme-f16f16}}
  svmls_lane_za16_f16_vg1x4(slice, zn4, zm, 7);

  // expected-error@+1 {{'svmla_single_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svmla_single_za16_bf16_vg1x2(slice, bzn2, bzm);
  // expected-error@+1 {{'svmla_single_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svmla_single_za16_bf16_vg1x4(slice, bzn4, bzm);
  // expected-error@+1 {{'svmls_single_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svmls_single_za16_bf16_vg1x2(slice, bzn2, bzm);
  // expected-error@+1 {{'svmls_single_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svmls_single_za16_bf16_vg1x4(slice, bzn4, bzm);
  // expected-error@+1 {{'svmla_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svmla_za16_bf16_vg1x2(slice, bzn2, bzm2);
  // expected-error@+1 {{'svmla_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svmla_za16_bf16_vg1x4(slice, bzn4, bzm4);
  // expected-error@+1 {{'svmls_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svmls_za16_bf16_vg1x2(slice, bzn2, bzm2);
  // expected-error@+1 {{'svmls_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svmls_za16_bf16_vg1x4(slice, bzn4, bzm4);
  // expected-error@+1 {{'svmla_lane_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svmla_lane_za16_bf16_vg1x2(slice, bzn2, bzm, 7);
  // expected-error@+1 {{'svmla_lane_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svmla_lane_za16_bf16_vg1x4(slice, bzn4, bzm, 7);
  // expected-error@+1 {{'svmls_lane_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svmls_lane_za16_bf16_vg1x2(slice, bzn2, bzm, 7);
  // expected-error@+1 {{'svmls_lane_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svmls_lane_za16_bf16_vg1x4(slice, bzn4, bzm, 7);
}


void test_imm(uint32_t slice, svfloat16_t zm, svfloat16x2_t zn2,svfloat16x4_t zn4,
              svbfloat16_t bzm, svbfloat16x2_t bzn2, svbfloat16x4_t bzn4)
  __arm_streaming __arm_inout("za") {

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmla_lane_za16_f16_vg1x2(slice, zn2, zm, -1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmla_lane_za16_f16_vg1x4(slice, zn4, zm, -1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmls_lane_za16_f16_vg1x2(slice, zn2, zm, -1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmls_lane_za16_f16_vg1x4(slice, zn4, zm, -1);

  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmla_lane_za16_bf16_vg1x2(slice, bzn2, bzm, -1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmla_lane_za16_bf16_vg1x4(slice, bzn4, bzm, -1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmls_lane_za16_bf16_vg1x2(slice, bzn2, bzm, -1);
  // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svmls_lane_za16_bf16_vg1x4(slice, bzn4, bzm, -1);
}
