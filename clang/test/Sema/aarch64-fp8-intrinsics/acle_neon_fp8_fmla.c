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

