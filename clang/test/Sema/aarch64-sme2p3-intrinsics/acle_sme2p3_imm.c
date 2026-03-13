// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -target-feature +sme2p3 -target-feature +bf16 -fsyntax-only -verify %s

#include <arm_sme.h>

void test_range_0_0(void) __arm_streaming __arm_in("zt0") {
  svluti6_zt_s8(1, svundef_u8()); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svluti6_zt_u8_x4(1, svcreate3_u8(svundef_u8(), svundef_u8(), svundef_u8())); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}

void test_range_0_1(void) __arm_streaming {
  svluti6_lane_s16_x4(svcreate2_s16(svundef_s16(), svundef_s16()), // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
                      svcreate2_u8(svundef_u8(), svundef_u8()), -1);
  svluti6_lane_u16_x4(svcreate2_u16(svundef_u16(), svundef_u16()), // expected-error {{argument value 2 is outside the valid range [0, 1]}}
                      svcreate2_u8(svundef_u8(), svundef_u8()), 2);
  svluti6_lane_f16_x4(svcreate2_f16(svundef_f16(), svundef_f16()), // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
                      svcreate2_u8(svundef_u8(), svundef_u8()), -1);
  svluti6_lane_bf16_x4(svcreate2_bf16(svundef_bf16(), svundef_bf16()), // expected-error {{argument value 2 is outside the valid range [0, 1]}}
                       svcreate2_u8(svundef_u8(), svundef_u8()), 2);
}
