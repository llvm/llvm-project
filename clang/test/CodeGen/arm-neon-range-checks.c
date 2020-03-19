// RUN: %clang_cc1 -triple arm64-none-eabi -target-feature +neon -target-feature +dotprod -target-feature +v8.1a -verify %s
// RUN: %clang_cc1 -triple armv8.1a-none-eabi -target-feature +neon -target-feature +dotprod -target-feature +v8.1a -verify %s

#include <arm_neon.h>

void test_vdot_lane(int32x2_t r, int8x8_t a, int8x8_t b) {
  vdot_lane_s32(r, a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vdot_lane_s32(r, a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vdot_lane_s32(r, a, b, 0);
  vdot_lane_s32(r, a, b, 1);
}

void test_vdotq_lane(int32x4_t r, int8x16_t a, int8x8_t b) {
  vdotq_lane_s32(r, a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vdotq_lane_s32(r, a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vdotq_lane_s32(r, a, b, 0);
  vdotq_lane_s32(r, a, b, 1);
}

#if defined(__aarch64__)
void test_vdot_laneq(int32x2_t r, int8x8_t a, int8x16_t b) {
  vdot_laneq_s32(r, a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vdot_laneq_s32(r, a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vdot_laneq_s32(r, a, b, 0);
  vdot_laneq_s32(r, a, b, 3);
}

void test_vdotq_laneq(int32x4_t r, int8x16_t a, int8x16_t b) {
  vdotq_laneq_s32(r, a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vdotq_laneq_s32(r, a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vdotq_laneq_s32(r, a, b, 0);
  vdotq_laneq_s32(r, a, b, 3);
}
#endif

void test_vdup_lane(int32x2_t v) {
  vdup_lane_s32(v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vdup_lane_s32(v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vdup_lane_s32(v, 0);
  vdup_lane_s32(v, 1);
}

void test_vdupq_lane(int32x2_t v) {
  vdupq_lane_s32(v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vdupq_lane_s32(v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vdupq_lane_s32(v, 0);
  vdupq_lane_s32(v, 1);
}

#if defined(__aarch64__)
void test_vdup_laneq(int32x4_t v) {
  vdup_laneq_s32(v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vdup_laneq_s32(v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vdup_laneq_s32(v, 0);
  vdup_laneq_s32(v, 3);
}

void test_vdupq_laneq(int32x4_t v) {
  vdupq_laneq_s32(v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vdupq_laneq_s32(v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vdupq_laneq_s32(v, 0);
  vdupq_laneq_s32(v, 3);
}
#endif

void test_vmla_lane(int32x2_t a, int32x2_t b, int32x2_t v) {
  vmla_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmla_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmla_lane_s32(a, b, v, 0);
  vmla_lane_s32(a, b, v, 1);
}

void test_vmlaq_lane(int32x4_t a, int32x4_t b, int32x2_t v) {
  vmlaq_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmlaq_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmlaq_lane_s32(a, b, v, 0);
  vmlaq_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vmla_laneq(int32x2_t a, int32x2_t b, int32x4_t v) {
  vmla_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmla_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmla_laneq_s32(a, b, v, 0);
  vmla_laneq_s32(a, b, v, 3);
}

void test_vmlaq_laneq(int32x4_t a, int32x4_t b, int32x4_t v) {
  vmlaq_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmlaq_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmlaq_laneq_s32(a, b, v, 0);
  vmlaq_laneq_s32(a, b, v, 3);
}

void test_vmlal_high_lane(int64x2_t a, int32x4_t b, int32x2_t v) {
  vmlal_high_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmlal_high_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmlal_high_lane_s32(a, b, v, 0);
  vmlal_high_lane_s32(a, b, v, 1);
}

void test_vmlal_high_laneq(int64x2_t a, int32x4_t b, int32x4_t v) {
  vmlal_high_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmlal_high_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmlal_high_laneq_s32(a, b, v, 0);
  vmlal_high_laneq_s32(a, b, v, 3);
}
#endif

void test_vmlal_lane(int64x2_t a, int32x2_t b, int32x2_t v) {
  vmlal_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmlal_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmlal_lane_s32(a, b, v, 0);
  vmlal_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vmlal_laneq(int64x2_t a, int32x2_t b, int32x4_t v) {
  vmlal_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmlal_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmlal_laneq_s32(a, b, v, 0);
  vmlal_laneq_s32(a, b, v, 3);
}
#endif

void test_vmls_lane(int32x2_t a, int32x2_t b, int32x2_t v) {
  vmls_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmls_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmls_lane_s32(a, b, v, 0);
  vmls_lane_s32(a, b, v, 1);
}

void test_vmlsq_lane(int32x4_t a, int32x4_t b, int32x2_t v) {
  vmlsq_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmlsq_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmlsq_lane_s32(a, b, v, 0);
  vmlsq_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vmls_laneq(int32x2_t a, int32x2_t b, int32x4_t v) {
  vmls_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmls_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmls_laneq_s32(a, b, v, 0);
  vmls_laneq_s32(a, b, v, 3);
}

void test_vmlsq_laneq(int32x4_t a, int32x4_t b, int32x4_t v) {
  vmlsq_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmlsq_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmlsq_laneq_s32(a, b, v, 0);
  vmlsq_laneq_s32(a, b, v, 3);
}

void test_vmlsl_high_lane(int64x2_t a, int32x4_t b, int32x2_t v) {
  vmlsl_high_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmlsl_high_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmlsl_high_lane_s32(a, b, v, 0);
  vmlsl_high_lane_s32(a, b, v, 1);
}

void test_vmlsl_high_laneq(int64x2_t a, int32x4_t b, int32x4_t v) {
  vmlsl_high_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmlsl_high_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmlsl_high_laneq_s32(a, b, v, 0);
  vmlsl_high_laneq_s32(a, b, v, 3);
}
#endif

void test_vmlsl_lane(int64x2_t a, int32x2_t b, int32x2_t v) {
  vmlsl_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmlsl_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmlsl_lane_s32(a, b, v, 0);
  vmlsl_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vmlsl_laneq(int64x2_t a, int32x2_t b, int32x4_t v) {
  vmlsl_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmlsl_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmlsl_laneq_s32(a, b, v, 0);
  vmlsl_laneq_s32(a, b, v, 3);
}
#endif

void test_vmull_lane(int32x2_t a, int32x2_t b) {
  vmull_lane_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmull_lane_s32(a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmull_lane_s32(a, b, 0);
  vmull_lane_s32(a, b, 1);
}

#if defined(__aarch64__)
void test_vmull_laneq(int32x2_t a, int32x4_t b) {
  vmull_laneq_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmull_laneq_s32(a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmull_laneq_s32(a, b, 0);
  vmull_laneq_s32(a, b, 3);
}

void test_vmull_high_lane(int32x4_t a, int32x2_t b) {
  vmull_high_lane_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vmull_high_lane_s32(a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vmull_high_lane_s32(a, b, 0);
  vmull_high_lane_s32(a, b, 1);
}

void test_vmull_high_laneq(int32x4_t a, int32x4_t b) {
  vmull_high_laneq_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vmull_high_laneq_s32(a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vmull_high_laneq_s32(a, b, 0);
  vmull_high_laneq_s32(a, b, 3);
}

void test_vqdmlal_high_lane(int64x2_t a, int32x4_t b, int32x2_t v) {
  vqdmlal_high_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmlal_high_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmlal_high_lane_s32(a, b, v, 0);
  vqdmlal_high_lane_s32(a, b, v, 1);
}

void test_vqdmlal_high_laneq(int64x2_t a, int32x4_t b, int32x4_t v) {
  vqdmlal_high_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmlal_high_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmlal_high_laneq_s32(a, b, v, 0);
  vqdmlal_high_laneq_s32(a, b, v, 3);
}
#endif

void test_vqdmlal_lane(int64x2_t a, int32x2_t b, int32x2_t v) {
  vqdmlal_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmlal_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmlal_lane_s32(a, b, v, 0);
  vqdmlal_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vqdmlal_laneq(int64x2_t a, int32x2_t b, int32x4_t v) {
  vqdmlal_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmlal_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmlal_laneq_s32(a, b, v, 0);
  vqdmlal_laneq_s32(a, b, v, 3);
}

void test_vqdmlsl_high_lane(int64x2_t a, int32x4_t b, int32x2_t v) {
  vqdmlsl_high_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmlsl_high_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmlsl_high_lane_s32(a, b, v, 0);
  vqdmlsl_high_lane_s32(a, b, v, 1);
}

void test_vqdmlsl_high_laneq(int64x2_t a, int32x4_t b, int32x4_t v) {
  vqdmlsl_high_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmlsl_high_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmlsl_high_laneq_s32(a, b, v, 0);
  vqdmlsl_high_laneq_s32(a, b, v, 3);
}
#endif

void test_vqdmlsl_lane(int64x2_t a, int32x2_t b, int32x2_t v) {
  vqdmlsl_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmlsl_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmlsl_lane_s32(a, b, v, 0);
  vqdmlsl_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vqdmlsl_laneq(int64x2_t a, int32x2_t b, int32x4_t v) {
  vqdmlsl_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmlsl_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmlsl_laneq_s32(a, b, v, 0);
  vqdmlsl_laneq_s32(a, b, v, 3);
}
#endif

void test_vqdmulh_lane(int32x2_t a, int32x2_t b) {
  vqdmulh_lane_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmulh_lane_s32(a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmulh_lane_s32(a, b, 0);
  vqdmulh_lane_s32(a, b, 1);
}

void test_vqdmulhq_lane(int32x4_t a, int32x2_t b) {
  vqdmulhq_lane_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmulhq_lane_s32(a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmulhq_lane_s32(a, b, 0);
  vqdmulhq_lane_s32(a, b, 1);
}

#if defined(__aarch64__)
void test_vqdmulh_laneq(int32x2_t a, int32x4_t b) {
  vqdmulh_laneq_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmulh_laneq_s32(a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmulh_laneq_s32(a, b, 0);
  vqdmulh_laneq_s32(a, b, 3);
}

void test_vqdmulhq_laneq(int32x4_t a, int32x4_t b) {
  vqdmulhq_laneq_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmulhq_laneq_s32(a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmulhq_laneq_s32(a, b, 0);
  vqdmulhq_laneq_s32(a, b, 3);
}

void test_vqdmull_high_lane(int32x4_t a, int32x2_t b) {
  vqdmull_high_lane_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmull_high_lane_s32(a, b, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmull_high_lane_s32(a, b, 0);
  vqdmull_high_lane_s32(a, b, 1);
}

void test_vqdmull_high_laneq(int32x4_t a, int32x4_t b) {
  vqdmull_high_laneq_s32(a, b, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmull_high_laneq_s32(a, b, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmull_high_laneq_s32(a, b, 0);
  vqdmull_high_laneq_s32(a, b, 3);
}
#endif

void test_vqdmull_lane(int32x2_t a, int32x2_t v) {
  vqdmull_lane_s32(a, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqdmull_lane_s32(a, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqdmull_lane_s32(a, v, 0);
  vqdmull_lane_s32(a, v, 1);
}

#if defined(__aarch64__)
void test_vqdmull_laneq(int32x2_t a, int32x4_t v) {
  vqdmull_laneq_s32(a, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqdmull_laneq_s32(a, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqdmull_laneq_s32(a, v, 0);
  vqdmull_laneq_s32(a, v, 3);
}
#endif

void test_vqrdmlah_lane(int32x2_t a, int32x2_t b, int32x2_t v) {
  vqrdmlah_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqrdmlah_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqrdmlah_lane_s32(a, b, v, 0);
  vqrdmlah_lane_s32(a, b, v, 1);
}

void test_vqrdmlahq_lane(int32x4_t a, int32x4_t b, int32x2_t v) {
  vqrdmlahq_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqrdmlahq_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqrdmlahq_lane_s32(a, b, v, 0);
  vqrdmlahq_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vqrdmlah_laneq(int32x2_t a, int32x2_t b, int32x4_t v) {
  vqrdmlah_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqrdmlah_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqrdmlah_laneq_s32(a, b, v, 0);
  vqrdmlah_laneq_s32(a, b, v, 3);
}

void test_vqrdmlahq_laneq(int32x4_t a, int32x4_t b, int32x4_t v) {
  vqrdmlahq_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqrdmlahq_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqrdmlahq_laneq_s32(a, b, v, 0);
  vqrdmlahq_laneq_s32(a, b, v, 3);
}
#endif

void test_vqrdmlsh_lane(int32x2_t a, int32x2_t b, int32x2_t v) {
  vqrdmlsh_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqrdmlsh_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqrdmlsh_lane_s32(a, b, v, 0);
  vqrdmlsh_lane_s32(a, b, v, 1);
}

void test_vqrdmlshq_lane(int32x4_t a, int32x4_t b, int32x2_t v) {
  vqrdmlshq_lane_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqrdmlshq_lane_s32(a, b, v, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqrdmlshq_lane_s32(a, b, v, 0);
  vqrdmlshq_lane_s32(a, b, v, 1);
}

#if defined(__aarch64__)
void test_vqrdmlsh_laneq(int32x2_t a, int32x2_t b, int32x4_t v) {
  vqrdmlsh_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqrdmlsh_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqrdmlsh_laneq_s32(a, b, v, 0);
  vqrdmlsh_laneq_s32(a, b, v, 3);
}

void test_vqrdmlshq_laneq(int32x4_t a, int32x4_t b, int32x4_t v) {
  vqrdmlshq_laneq_s32(a, b, v, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqrdmlshq_laneq_s32(a, b, v, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqrdmlshq_laneq_s32(a, b, v, 0);
  vqrdmlshq_laneq_s32(a, b, v, 3);
}
#endif

void test_vqrdmulh_lane(int32x2_t a, int32x2_t v) {
  vqrdmulh_lane_s32(a, v,  -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqrdmulh_lane_s32(a, v,  2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqrdmulh_lane_s32(a, v,  0);
  vqrdmulh_lane_s32(a, v,  1);
}

void test_vqrdmulhq_lane(int32x4_t a, int32x2_t v) {
  vqrdmulhq_lane_s32(a, v,  -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vqrdmulhq_lane_s32(a, v,  2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  vqrdmulhq_lane_s32(a, v,  0);
  vqrdmulhq_lane_s32(a, v,  1);
}

#if defined(__aarch64__)
void test_vqrdmulh_laneq(int32x2_t a, int32x4_t v) {
  vqrdmulh_laneq_s32(a, v,  -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqrdmulh_laneq_s32(a, v,  4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqrdmulh_laneq_s32(a, v,  0);
  vqrdmulh_laneq_s32(a, v,  3);
}

void test_vqrdmulhq_laneq(int32x4_t a, int32x4_t v) {
  vqrdmulhq_laneq_s32(a, v,  -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vqrdmulhq_laneq_s32(a, v,  4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  vqrdmulhq_laneq_s32(a, v,  0);
  vqrdmulhq_laneq_s32(a, v,  3);
}
#endif
