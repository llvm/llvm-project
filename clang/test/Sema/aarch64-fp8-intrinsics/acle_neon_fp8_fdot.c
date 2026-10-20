// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +bf16 -target-feature +faminmax -target-feature +fp8 -emit-llvm -verify %s -o /dev/null

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

void test_features(float16x4_t vd4, float16x8_t vd8, float32x4_t va4, float32x2_t va2,
                   mfloat8x8_t v8, mfloat8x16_t v16, fpm_t fpm) {
  (void) vdot_f16_mf8_fpm(vd4, v8, v8, fpm);
// expected-error@-1 {{'vdot_f16_mf8_fpm' requires target feature 'fp8dot2'}}
  (void) vdotq_f16_mf8_fpm(vd8, v16, v16, fpm);
// expected-error@-1 {{'vdotq_f16_mf8_fpm' requires target feature 'fp8dot2'}}
  (void) vdot_lane_f16_mf8_fpm(vd4, v8, v8, 3, fpm);
// expected-error@-1 {{'__builtin_neon_vdot_lane_f16_mf8_fpm' needs target feature fp8dot2,neon}}
  (void) vdot_laneq_f16_mf8_fpm(vd4, v8, v16, 7, fpm);
// expected-error@-1 {{'__builtin_neon_vdot_laneq_f16_mf8_fpm' needs target feature fp8dot2,neon}}
  (void) vdotq_lane_f16_mf8_fpm(vd8, v16, v8, 3, fpm);
// expected-error@-1 {{'__builtin_neon_vdotq_lane_f16_mf8_fpm' needs target feature fp8dot2,neon}}
  (void) vdotq_laneq_f16_mf8_fpm(vd8, v16, v16, 7, fpm);
// expected-error@-1 {{'__builtin_neon_vdotq_laneq_f16_mf8_fpm' needs target feature fp8dot2,neon}}

  (void) vdot_f32_mf8_fpm(va2, v8, v8, fpm);
// expected-error@-1 {{'vdot_f32_mf8_fpm' requires target feature 'fp8dot4'}}
  (void) vdotq_f32_mf8_fpm(va4, v16, v16, fpm);
// expected-error@-1 {{'vdotq_f32_mf8_fpm' requires target feature 'fp8dot4}}
  (void) vdot_lane_f32_mf8_fpm(va2, v8, v8, 1, fpm);
// expected-error@-1 {{'__builtin_neon_vdot_lane_f32_mf8_fpm' needs target feature fp8dot4,neon}}
  (void) vdot_laneq_f32_mf8_fpm(va2, v8, v16, 3, fpm);
// expected-error@-1 {{'__builtin_neon_vdot_laneq_f32_mf8_fpm' needs target feature fp8dot4,neon}}
  (void) vdotq_lane_f32_mf8_fpm(va4, v16, v8, 1, fpm);
// expected-error@-1 {{'__builtin_neon_vdotq_lane_f32_mf8_fpm' needs target feature fp8dot4,neon}}
  (void) vdotq_laneq_f32_mf8_fpm(va4, v16, v16, 3, fpm);
// expected-error@-1 {{'__builtin_neon_vdotq_laneq_f32_mf8_fpm' needs target feature fp8dot4,neon}}
}

void test_imm(float16x4_t vd4, float16x8_t vd8, float32x2_t va2, float32x4_t va4,
              mfloat8x8_t v8, mfloat8x16_t v16, fpm_t fpm) {
  (void) vdot_lane_f16_mf8_fpm(vd4, v8, v8, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 3]}}
  (void) vdot_laneq_f16_mf8_fpm(vd4, v8, v16, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
  (void) vdotq_lane_f16_mf8_fpm(vd8, v16, v8, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 3]}}
  (void) vdotq_laneq_f16_mf8_fpm(vd8, v16, v16, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 7]}}
  (void) vdot_lane_f32_mf8_fpm(va2, v8, v8, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 1]}}
  (void) vdot_laneq_f32_mf8_fpm(va2, v8, v16, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 3]}}
  (void) vdotq_lane_f32_mf8_fpm(va4, v16, v8, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 1]}}
  (void) vdotq_laneq_f32_mf8_fpm(va4, v16, v16, -1, fpm);
  // expected-error@-1 {{argument value -1 is outside the valid range [0, 3]}}
}
