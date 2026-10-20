// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -verify -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features(uint32_t slice, svmfloat8_t f8, svmfloat8x2_t f8x2,
                   svmfloat8x4_t f8x4, uint64_t fpmr) __arm_streaming __arm_inout("za") {
  // expected-error@+1 {{'svdot_lane_za32_mf8_vg1x2_fpm' needs target feature sme,sme-f8f32}}
  svdot_lane_za32_mf8_vg1x2_fpm(slice, f8x2, f8, 3, fpmr);
  // expected-error@+1 {{'svdot_lane_za32_mf8_vg1x4_fpm' needs target feature sme,sme-f8f32}}
  svdot_lane_za32_mf8_vg1x4_fpm(slice, f8x4, f8, 3, fpmr);
  // expected-error@+1 {{'svdot_lane_za16_mf8_vg1x2_fpm' needs target feature sme,sme-f8f16}}
  svdot_lane_za16_mf8_vg1x2_fpm(slice, f8x2, f8, 3, fpmr);
  // expected-error@+1 {{'svdot_lane_za16_mf8_vg1x4_fpm' needs target feature sme,sme-f8f16}}
  svdot_lane_za16_mf8_vg1x4_fpm(slice, f8x4, f8, 3, fpmr);
  // expected-error@+1 {{'svdot_single_za32_mf8_vg1x2_fpm' needs target feature sme,sme-f8f32}}
  svdot_single_za32_mf8_vg1x2_fpm(slice, f8x2, f8, fpmr);
  // expected-error@+1 {{'svdot_single_za32_mf8_vg1x4_fpm' needs target feature sme,sme-f8f32}}
  svdot_single_za32_mf8_vg1x4_fpm(slice, f8x4, f8, fpmr);
  // expected-error@+1 {{'svdot_za32_mf8_vg1x2_fpm' needs target feature sme,sme-f8f32}}
  svdot_za32_mf8_vg1x2_fpm(slice, f8x2, f8x2, fpmr);
  // expected-error@+1 {{'svdot_za32_mf8_vg1x4_fpm' needs target feature sme,sme-f8f32}}
  svdot_za32_mf8_vg1x4_fpm(slice, f8x4, f8x4, fpmr);
  // expected-error@+1 {{'svdot_single_za16_mf8_vg1x2_fpm' needs target feature sme,sme-f8f16}}
  svdot_single_za16_mf8_vg1x2_fpm(slice, f8x2, f8, fpmr);
  // expected-error@+1 {{'svdot_single_za16_mf8_vg1x4_fpm' needs target feature sme,sme-f8f16}}
  svdot_single_za16_mf8_vg1x4_fpm(slice, f8x4, f8, fpmr);
  // expected-error@+1 {{'svdot_za16_mf8_vg1x2_fpm' needs target feature sme,sme-f8f16}}
  svdot_za16_mf8_vg1x2_fpm(slice, f8x2, f8x2, fpmr);
  // expected-error@+1 {{'svdot_za16_mf8_vg1x4_fpm' needs target feature sme,sme-f8f16}}
  svdot_za16_mf8_vg1x4_fpm(slice, f8x4, f8x4, fpmr);
}

void test_imm(uint32_t slice, svmfloat8_t f8, svmfloat8x2_t f8x2,
              svmfloat8x4_t f8x4, uint64_t fpmr) __arm_streaming __arm_inout("za") {
// expected-error@+1{{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svdot_lane_za32_mf8_vg1x2_fpm(slice, f8x2, f8, -1, fpmr);
// expected-error@+1{{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svdot_lane_za32_mf8_vg1x4_fpm(slice, f8x4, f8, -1, fpmr);
// expected-error@+1{{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svdot_lane_za16_mf8_vg1x2_fpm(slice, f8x2, f8, -1, fpmr);
// expected-error@+1{{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svdot_lane_za16_mf8_vg1x4_fpm(slice, f8x4, f8, -1, fpmr);

// expected-error@+1{{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_mf8_vg1x2_fpm(slice, f8x2, f8, 4, fpmr);
// expected-error@+1{{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_mf8_vg1x4_fpm(slice, f8x4, f8, 4, fpmr);
// expected-error@+1{{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za16_mf8_vg1x2_fpm(slice, f8x2, f8, 8, fpmr);
// expected-error@+1{{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za16_mf8_vg1x4_fpm(slice, f8x4, f8, 8, fpmr);
}
