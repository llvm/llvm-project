// RUN: %clang_cc1 -triple arm64-apple-darwin -target-feature +neon -fsyntax-only -ffreestanding -verify %s
// RUN: %clang_cc1 -triple aarch64_be-none-linux-gnu -target-feature +neon -fsyntax-only -ffreestanding -verify %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// rdar://13527900
void vcopy_reject(float32x4_t vOut0, float32x4_t vAlpha, int t) {
  vcopyq_laneq_f32(vOut0, 1, vAlpha, t); // expected-error {{argument to '__builtin_neon_vgetq_lane_f32' must be a constant integer}}
}

// rdar://problem/15256199
float32x4_t test_vmlsq_lane(float32x4_t accum, float32x4_t lhs, float32x2_t rhs) {
  return vmlsq_lane_f32(accum, lhs, rhs, 1);
}
