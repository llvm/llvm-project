// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-feature +neon -target-feature +bf16 \
// RUN:  -O2 -fallow-half-arguments-and-returns -verify -fsyntax-only %s

#include "arm_neon.h"

int x;

bfloat16x4_t test_vld1_lane_bf16(bfloat16_t const *ptr, bfloat16x4_t src) {
  (void)vld1_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld1_lane_bf16(ptr, src, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld1_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x8_t test_vld1q_lane_bf16(bfloat16_t const *ptr, bfloat16x8_t src) {
  (void)vld1q_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld1q_lane_bf16(ptr, src, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld1q_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x4x2_t test_vld2_lane_bf16(bfloat16_t const *ptr, bfloat16x4x2_t src) {
  (void)vld2_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld2_lane_bf16(ptr, src, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld2_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x8x2_t test_vld2q_lane_bf16(bfloat16_t const *ptr, bfloat16x8x2_t src) {
  (void)vld2q_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld2q_lane_bf16(ptr, src, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld2q_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x4x3_t test_vld3_lane_bf16(bfloat16_t const *ptr, bfloat16x4x3_t src) {
  (void)vld3_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld3_lane_bf16(ptr, src, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld3_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x8x3_t test_vld3q_lane_bf16(bfloat16_t const *ptr, bfloat16x8x3_t src) {
  (void)vld3q_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld3q_lane_bf16(ptr, src, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld3q_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x4x4_t test_vld4_lane_bf16(bfloat16_t const *ptr, bfloat16x4x4_t src) {
  (void)vld4_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld4_lane_bf16(ptr, src, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld4_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

bfloat16x8x4_t test_vld4q_lane_bf16(bfloat16_t const *ptr, bfloat16x8x4_t src) {
  (void)vld4q_lane_bf16(ptr, src, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  (void)vld4q_lane_bf16(ptr, src, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  return vld4q_lane_bf16(ptr, src, x); // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst1_lane_bf16(bfloat16_t *ptr, bfloat16x4_t val) {
  vst1_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst1_lane_bf16(ptr, val, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst1_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst1q_lane_bf16(bfloat16_t *ptr, bfloat16x8_t val) {
  vst1q_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst1q_lane_bf16(ptr, val, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst1q_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst2_lane_bf16(bfloat16_t *ptr, bfloat16x4x2_t val) {
  vst2_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst2_lane_bf16(ptr, val, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst2_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst2q_lane_bf16(bfloat16_t *ptr, bfloat16x8x2_t val) {
  vst2q_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst2q_lane_bf16(ptr, val, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst2q_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst3_lane_bf16(bfloat16_t *ptr, bfloat16x4x3_t val) {
  vst3_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst3_lane_bf16(ptr, val, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst3_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst3q_lane_bf16(bfloat16_t *ptr, bfloat16x8x3_t val) {
  vst3q_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst3q_lane_bf16(ptr, val, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst3q_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst4_lane_bf16(bfloat16_t *ptr, bfloat16x4x4_t val) {
  vst4_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst4_lane_bf16(ptr, val, 4);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst4_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}

void test_vst4q_lane_bf16(bfloat16_t *ptr, bfloat16x8x4_t val) {
  vst4q_lane_bf16(ptr, val, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst4q_lane_bf16(ptr, val, 8);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vst4q_lane_bf16(ptr, val, x);  // expected-error-re {{argument {{.*}} must be a constant integer}}
}
