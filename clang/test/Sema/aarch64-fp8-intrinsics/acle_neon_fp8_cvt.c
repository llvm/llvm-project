// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +bf16 -target-feature +faminmax -emit-llvm -verify %s -o /dev/null

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

void test_features(float16x4_t vd4, float16x8_t vd8, float32x4_t va4,
                   mfloat8x8_t v8, mfloat8x16_t v16, fpm_t fpm) {
  (void) vcvt1_bf16_mf8_fpm(v8, fpm);
  // expected-error@-1 {{'vcvt1_bf16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt1_low_bf16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt1_low_bf16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt2_bf16_mf8_fpm(v8, fpm);
  // expected-error@-1 {{'vcvt2_bf16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt2_low_bf16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt2_low_bf16_mf8_fpm' requires target feature 'fp8'}}

  (void) vcvt1_high_bf16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt1_high_bf16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt2_high_bf16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt2_high_bf16_mf8_fpm' requires target feature 'fp8'}}

  (void) vcvt1_f16_mf8_fpm(v8, fpm);
  // expected-error@-1 {{'vcvt1_f16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt1_low_f16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt1_low_f16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt2_f16_mf8_fpm(v8, fpm);
  // expected-error@-1 {{'vcvt2_f16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt2_low_f16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt2_low_f16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt1_high_f16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt1_high_f16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt2_high_f16_mf8_fpm(v16, fpm);
  // expected-error@-1 {{'vcvt2_high_f16_mf8_fpm' requires target feature 'fp8'}}
  (void) vcvt_mf8_f32_fpm(va4, va4, fpm);
  // expected-error@-1 {{'vcvt_mf8_f32_fpm' requires target feature 'fp8'}}
  (void) vcvt_high_mf8_f32_fpm(v8, va4, va4, fpm);
  // expected-error@-1 {{'vcvt_high_mf8_f32_fpm' requires target feature 'fp8'}}
  (void) vcvt_mf8_f16_fpm(vd4, vd4, fpm);
  // expected-error@-1 {{'vcvt_mf8_f16_fpm' requires target feature 'fp8'}}
  (void) vcvtq_mf8_f16_fpm(vd8, vd8, fpm);
  // expected-error@-1 {{'vcvtq_mf8_f16_fpm' requires target feature 'fp8'}}
}
