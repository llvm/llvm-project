// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +bf16 -target-feature +faminmax -target-feature +fp8 -emit-llvm -verify %s -o /dev/null

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

void test_features(float16x8_t a, float32x4_t b, mfloat8x16_t u, fpm_t fpm) {
  (void) vmlalbq_f16_mf8_fpm(a, u, u, fpm);
  // expected-error@-1 {{'vmlalbq_f16_mf8_fpm' requires target feature 'fp8fma'}}
  (void) vmlaltq_f16_mf8_fpm(a, u, u, fpm);
  // expected-error@-1 {{'vmlaltq_f16_mf8_fpm' requires target feature 'fp8fma'}}
  (void) vmlallbbq_f32_mf8_fpm(b, u, u, fpm);
  // expected-error@-1 {{'vmlallbbq_f32_mf8_fpm' requires target feature 'fp8fma'}}
  (void) vmlallbtq_f32_mf8_fpm(b, u, u, fpm);
  // expected-error@-1 {{'vmlallbtq_f32_mf8_fpm' requires target feature 'fp8fma'}}
  (void) vmlalltbq_f32_mf8_fpm(b, u, u, fpm);
  // expected-error@-1 {{'vmlalltbq_f32_mf8_fpm' requires target feature 'fp8fma'}}
  (void) vmlallttq_f32_mf8_fpm(b, u, u, fpm);
  // expected-error@-1 {{'vmlallttq_f32_mf8_fpm' requires target feature 'fp8fma'}}
}

void test_imm(float16x8_t d, float32x4_t c, mfloat8x16_t a, mfloat8x8_t b, fpm_t fpm) {
(void) vmlalbq_lane_f16_mf8_fpm(d, a, b, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
(void) vmlalbq_laneq_f16_mf8_fpm(d, a, a, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 15]}}
(void) vmlaltq_lane_f16_mf8_fpm(d, a, b, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
(void) vmlaltq_laneq_f16_mf8_fpm(d, a, a, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 15]}}

(void) vmlallbbq_lane_f32_mf8_fpm(c, a, b, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
(void) vmlallbbq_laneq_f32_mf8_fpm(c, a, a, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 15]}}
(void) vmlallbtq_lane_f32_mf8_fpm(c, a, b, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
(void) vmlallbtq_laneq_f32_mf8_fpm(c, a, a, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 15]}}
(void) vmlalltbq_lane_f32_mf8_fpm(c, a, b, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
(void) vmlalltbq_laneq_f32_mf8_fpm(c, a, a, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 15]}}
(void) vmlallttq_lane_f32_mf8_fpm(c, a, b, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
(void) vmlallttq_laneq_f32_mf8_fpm(c, a, a, -1, fpm);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 15]}}
}

