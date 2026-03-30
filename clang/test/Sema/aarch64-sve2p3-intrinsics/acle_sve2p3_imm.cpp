// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2 -target-feature +sve2p3 -fsyntax-only -verify %s

#include <arm_sve.h>

void test_svdot_lane_x2_imm_0_7(svint16_t s16, svuint16_t u16, svint8_t s8,
                                svuint8_t u8) {
  svdot_lane_s16_s8(s16, s8, s8, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svdot_lane_u16_u8(u16, u8, u8, -1); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 7]}}

  svdot_lane_s16_s8(s16, s8, s8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_u16_u8(u16, u8, u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}
