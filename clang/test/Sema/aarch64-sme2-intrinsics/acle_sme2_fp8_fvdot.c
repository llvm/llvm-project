// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -verify -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features(uint32_t slice, fpm_t fpmr, svmfloat8x2_t zn,
                   svmfloat8_t zm) __arm_streaming __arm_inout("za") {
// expected-error@+1 {{'svvdot_lane_za16_f8_vg1x2' needs target feature sme,sme-f8f16}}
  svvdot_lane_za16_f8_vg1x2(slice, zn, zm, 7, fpmr);
// expected-error@+1 {{'svvdotb_lane_za32_f8_vg1x4' needs target feature sme,sme-f8f32}}
  svvdotb_lane_za32_f8_vg1x4(slice, zn, zm, 3, fpmr);
// expected-error@+1 {{'svvdott_lane_za32_f8_vg1x4' needs target feature sme,sme-f8f32}}
  svvdott_lane_za32_f8_vg1x4(slice, zn, zm, 3, fpmr);
}

void test_imm(uint32_t slice, fpm_t fpmr, svmfloat8x2_t zn,
              svmfloat8_t zm) __arm_streaming __arm_inout("za") {
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svvdot_lane_za16_f8_vg1x2(slice, zn, zm, -1, fpmr);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svvdotb_lane_za32_f8_vg1x4(slice, zn, zm, -1, fpmr);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svvdott_lane_za32_f8_vg1x4(slice, zn, zm, -1, fpmr);

// expected-error@+1{{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za16_f8_vg1x2(slice, zn, zm, 8, fpmr);
// expected-error@+1{{argument value 4 is outside the valid range [0, 3]}}
  svvdotb_lane_za32_f8_vg1x4(slice, zn, zm, 4, fpmr);
// expected-error@+1{{argument value 4 is outside the valid range [0, 3]}}
  svvdott_lane_za32_f8_vg1x4(slice, zn, zm, 4, fpmr);
}
