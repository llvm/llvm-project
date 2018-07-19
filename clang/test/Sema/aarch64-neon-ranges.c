// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>

void test_vext_8bit(int8x8_t small, int8x16_t big) {
  vext_s8(small, small, 7);
  vext_u8(small, small, 7);
  vext_p8(small, small, 7);
  vextq_s8(big, big, 15);
  vextq_u8(big, big, 15);
  vextq_p8(big, big, 15);

  vext_s8(small, small, 8); // expected-error {{argument should be a value from 0 to 7}}
  vext_u8(small, small, 8); // expected-error {{argument should be a value from 0 to 7}}
  vext_p8(small, small, 8); // expected-error {{argument should be a value from 0 to 7}}
  vextq_s8(big, big, 16); // expected-error {{argument should be a value from 0 to 15}}
  vextq_u8(big, big, 16); // expected-error {{argument should be a value from 0 to 15}}
  vextq_p8(big, big, 16); // expected-error {{argument should be a value from 0 to 15}}
}

void test_mul_lane_f64(float64x1_t small, float64x2_t big, float64x2_t rhs) {
  vmul_lane_f64(small, small, 0);
  vmul_laneq_f64(small, big, 1);
  vmulq_lane_f64(big, small, 0);
  vmulq_laneq_f64(big, big, 1);
  vfma_lane_f64(small, small, small, 0);
  vfma_laneq_f64(small, small, big, 1);
  vfmaq_lane_f64(big, big, small, 0);
  vfmaq_laneq_f64(big, big, big, 1);

  vmul_lane_f64(small, small, 1); // expected-error {{argument should be a value from 0 to 0}}
  vmul_laneq_f64(small, big, 2); // expected-error {{argument should be a value from 0 to 1}}
  vfma_lane_f64(small, small, small, 1); // expected-error {{argument should be a value from 0 to 0}}
  vfma_laneq_f64(small, small, big, 2); // expected-error {{argument should be a value from 0 to 1}}
  vfmaq_laneq_f64(big, big, big, 2); // expected-error {{argument should be a value from 0 to 1}}
}

void test_ld1st1(int8x8_t small, int8x16_t big, void *addr) {
  vld1_lane_s8(addr, small, 7);
  vld1_lane_s16(addr, small, 3);
  vld1_lane_s32(addr, small, 1);
  vld1_lane_s64(addr, small, 0);

  vld1q_lane_s8(addr, big, 15);
  vld1q_lane_s16(addr, big, 7);
  vld1q_lane_s32(addr, big, 3);
  vld1q_lane_s64(addr, big, 1);

  vld1_lane_s8(addr, small, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld1_lane_s16(addr, small, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld1_lane_s32(addr, small, 2); // expected-error {{argument should be a value from 0 to 1}}
  vld1_lane_s64(addr, small, 1); // expected-error {{argument should be a value from 0 to 0}}

  vld1q_lane_s8(addr, big, 16); // expected-error {{argument should be a value from 0 to 15}}
  vld1q_lane_s16(addr, big, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld1q_lane_s32(addr, big, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld1q_lane_s64(addr, big, 2); // expected-error {{argument should be a value from 0 to 1}}

  vst1_lane_s8(addr, small, 7);
  vst1_lane_s16(addr, small, 3);
  vst1_lane_s32(addr, small, 1);
  vst1_lane_s64(addr, small, 0);

  vst1q_lane_s8(addr, big, 15);
  vst1q_lane_s16(addr, big, 7);
  vst1q_lane_s32(addr, big, 3);
  vst1q_lane_s64(addr, big, 1);

  vst1_lane_s8(addr, small, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst1_lane_s16(addr, small, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst1_lane_s32(addr, small, 2); // expected-error {{argument should be a value from 0 to 1}}
  vst1_lane_s64(addr, small, 1); // expected-error {{argument should be a value from 0 to 0}}

  vst1q_lane_s8(addr, big, 16); // expected-error {{argument should be a value from 0 to 15}}
  vst1q_lane_s16(addr, big, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst1q_lane_s32(addr, big, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst1q_lane_s64(addr, big, 2); // expected-error {{argument should be a value from 0 to 1}}
}

void test_ld2st2(int8x8x2_t small8, int8x16x2_t big8,
                 int16x4x2_t small16, int16x8x2_t big16,
                 int32x2x2_t small32, int32x4x2_t big32,
                 int64x1x2_t small64, int64x2x2_t big64,
                 void *addr) {
  vld2_lane_s8(addr, small8, 7);
  vld2_lane_s16(addr, small16, 3);
  vld2_lane_s32(addr, small32, 1);
  vld2_lane_s64(addr, small64, 0);

  vld2q_lane_s8(addr, big8, 15);
  vld2q_lane_s16(addr, big16, 7);
  vld2q_lane_s32(addr, big32, 3);
  vld2q_lane_s64(addr, big64, 1);

  vld2_lane_s8(addr, small8, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld2_lane_s16(addr, small16, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld2_lane_s32(addr, small32, 2); // expected-error {{argument should be a value from 0 to 1}}
  vld2_lane_s64(addr, small64, 1); // expected-error {{argument should be a value from 0 to 0}}

  vld2q_lane_s8(addr, big8, 16); // expected-error {{argument should be a value from 0 to 15}}
  vld2q_lane_s16(addr, big16, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld2q_lane_s32(addr, big32, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld2q_lane_s64(addr, big64, 2); // expected-error {{argument should be a value from 0 to 1}}

  vst2_lane_s8(addr, small8, 7);
  vst2_lane_s16(addr, small16, 3);
  vst2_lane_s32(addr, small32, 1);
  vst2_lane_s64(addr, small64, 0);

  vst2q_lane_s8(addr, big8, 15);
  vst2q_lane_s16(addr, big16, 7);
  vst2q_lane_s32(addr, big32, 3);
  vst2q_lane_s64(addr, big64, 1);

  vst2_lane_s8(addr, small8, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst2_lane_s16(addr, small16, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst2_lane_s32(addr, small32, 2); // expected-error {{argument should be a value from 0 to 1}}
  vst2_lane_s64(addr, small64, 1); // expected-error {{argument should be a value from 0 to 0}}

  vst2q_lane_s8(addr, big8, 16); // expected-error {{argument should be a value from 0 to 15}}
  vst2q_lane_s16(addr, big16, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst2q_lane_s32(addr, big32, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst2q_lane_s64(addr, big64, 2); // expected-error {{argument should be a value from 0 to 1}}
}

void test_ld3st3(int8x8x3_t small8, int8x16x3_t big8,
                 int16x4x3_t small16, int16x8x3_t big16,
                 int32x2x3_t small32, int32x4x3_t big32,
                 int64x1x3_t small64, int64x2x3_t big64,
                 void *addr) {
  vld3_lane_s8(addr, small8, 7);
  vld3_lane_s16(addr, small16, 3);
  vld3_lane_s32(addr, small32, 1);
  vld3_lane_s64(addr, small64, 0);

  vld3q_lane_s8(addr, big8, 15);
  vld3q_lane_s16(addr, big16, 7);
  vld3q_lane_s32(addr, big32, 3);
  vld3q_lane_s64(addr, big64, 1);

  vld3_lane_s8(addr, small8, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld3_lane_s16(addr, small16, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld3_lane_s32(addr, small32, 2); // expected-error {{argument should be a value from 0 to 1}}
  vld3_lane_s64(addr, small64, 1); // expected-error {{argument should be a value from 0 to 0}}

  vld3q_lane_s8(addr, big8, 16); // expected-error {{argument should be a value from 0 to 15}}
  vld3q_lane_s16(addr, big16, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld3q_lane_s32(addr, big32, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld3q_lane_s64(addr, big64, 2); // expected-error {{argument should be a value from 0 to 1}}

  vst3_lane_s8(addr, small8, 7);
  vst3_lane_s16(addr, small16, 3);
  vst3_lane_s32(addr, small32, 1);
  vst3_lane_s64(addr, small64, 0);

  vst3q_lane_s8(addr, big8, 15);
  vst3q_lane_s16(addr, big16, 7);
  vst3q_lane_s32(addr, big32, 3);
  vst3q_lane_s64(addr, big64, 1);

  vst3_lane_s8(addr, small8, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst3_lane_s16(addr, small16, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst3_lane_s32(addr, small32, 2); // expected-error {{argument should be a value from 0 to 1}}
  vst3_lane_s64(addr, small64, 1); // expected-error {{argument should be a value from 0 to 0}}

  vst3q_lane_s8(addr, big8, 16); // expected-error {{argument should be a value from 0 to 15}}
  vst3q_lane_s16(addr, big16, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst3q_lane_s32(addr, big32, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst3q_lane_s64(addr, big64, 2); // expected-error {{argument should be a value from 0 to 1}}
}

void test_ld4st4(int8x8x4_t small8, int8x16x4_t big8,
                 int16x4x4_t small16, int16x8x4_t big16,
                 int32x2x4_t small32, int32x4x4_t big32,
                 int64x1x4_t small64, int64x2x4_t big64,
                 void *addr) {
  vld4_lane_s8(addr, small8, 7);
  vld4_lane_s16(addr, small16, 3);
  vld4_lane_s32(addr, small32, 1);
  vld4_lane_s64(addr, small64, 0);

  vld4q_lane_s8(addr, big8, 15);
  vld4q_lane_s16(addr, big16, 7);
  vld4q_lane_s32(addr, big32, 3);
  vld4q_lane_s64(addr, big64, 1);

  vld4_lane_s8(addr, small8, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld4_lane_s16(addr, small16, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld4_lane_s32(addr, small32, 2); // expected-error {{argument should be a value from 0 to 1}}
  vld4_lane_s64(addr, small64, 1); // expected-error {{argument should be a value from 0 to 0}}

  vld4q_lane_s8(addr, big8, 16); // expected-error {{argument should be a value from 0 to 15}}
  vld4q_lane_s16(addr, big16, 8); // expected-error {{argument should be a value from 0 to 7}}
  vld4q_lane_s32(addr, big32, 4); // expected-error {{argument should be a value from 0 to 3}}
  vld4q_lane_s64(addr, big64, 2); // expected-error {{argument should be a value from 0 to 1}}

  vst4_lane_s8(addr, small8, 7);
  vst4_lane_s16(addr, small16, 3);
  vst4_lane_s32(addr, small32, 1);
  vst4_lane_s64(addr, small64, 0);

  vst4q_lane_s8(addr, big8, 15);
  vst4q_lane_s16(addr, big16, 7);
  vst4q_lane_s32(addr, big32, 3);
  vst4q_lane_s64(addr, big64, 1);

  vst4_lane_s8(addr, small8, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst4_lane_s16(addr, small16, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst4_lane_s32(addr, small32, 2); // expected-error {{argument should be a value from 0 to 1}}
  vst4_lane_s64(addr, small64, 1); // expected-error {{argument should be a value from 0 to 0}}

  vst4q_lane_s8(addr, big8, 16); // expected-error {{argument should be a value from 0 to 15}}
  vst4q_lane_s16(addr, big16, 8); // expected-error {{argument should be a value from 0 to 7}}
  vst4q_lane_s32(addr, big32, 4); // expected-error {{argument should be a value from 0 to 3}}
  vst4q_lane_s64(addr, big64, 2); // expected-error {{argument should be a value from 0 to 1}}
}

