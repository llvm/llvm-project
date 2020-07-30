// RUN: %clang_cc1 -fsyntax-only -verify \
// RUN: -triple aarch64-arm-none-eabi -target-feature +neon \
// RUN: -target-feature +bf16 %s

#include <arm_neon.h>

int x;

void test_vcopy_lane_bf16(bfloat16x4_t a, bfloat16x8_t b) {
  // 0 <= lane1 <= 3; 0 <= lane2 <= 3
  (void)vcopy_lane_bf16(a, 3, a, 3);
  (void)vcopy_lane_bf16(a, 0, a, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  (void)vcopy_lane_bf16(a, 1, a, -1);   // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  (void)vcopy_lane_bf16(a, 4, a, 0);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  (void)vcopy_lane_bf16(a, -1, a, 1);   // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  (void)vcopy_lane_bf16(a, 0, a, x);    // expected-error-re {{argument {{.*}} must be a constant integer}}
  (void)vcopy_lane_bf16(a, x, a, 0);    // expected-error-re {{argument {{.*}} must be a constant integer}}

  // 0 <= lane1 <= 7; 0 <= lane2 <= 3
  (void)vcopyq_lane_bf16(b, 7, a, 3);
  (void)vcopyq_lane_bf16(b, 0, a, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  (void)vcopyq_lane_bf16(b, 1, a, -1);  // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  (void)vcopyq_lane_bf16(b, 8, a, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  (void)vcopyq_lane_bf16(b, -1, a, 1);  // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  (void)vcopyq_lane_bf16(b, 0, a, x);   // expected-error-re {{argument {{.*}} must be a constant integer}}
  (void)vcopyq_lane_bf16(b, x, a, 0);   // expected-error-re {{argument {{.*}} must be a constant integer}}

  // 0 <= lane1 <= 3; 0 <= lane2 <= 7
  (void)vcopy_laneq_bf16(a, 3, b, 7);
  (void)vcopy_laneq_bf16(a, 0, b, 8);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  (void)vcopy_laneq_bf16(a, 1, b, -1);  // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  (void)vcopy_laneq_bf16(a, 4, b, 0);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  (void)vcopy_laneq_bf16(a, -1, b, 1);  // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  (void)vcopy_laneq_bf16(a, 0, b, x);   // expected-error-re {{argument {{.*}} must be a constant integer}}
  (void)vcopy_laneq_bf16(a, x, b, 0);   // expected-error-re {{argument {{.*}} must be a constant integer}}


  // 0 <= lane1 <= 7; 0 <= lane2 <= 7
  (void)vcopyq_laneq_bf16(b, 7, b, 7);
  (void)vcopyq_laneq_bf16(b, 0, b, 8);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  (void)vcopyq_laneq_bf16(b, 1, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  (void)vcopyq_laneq_bf16(b, 8, b, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  (void)vcopyq_laneq_bf16(b, -1, b, 1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  (void)vcopyq_laneq_bf16(b, 0, b, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
  (void)vcopyq_laneq_bf16(b, x, b, 0);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

