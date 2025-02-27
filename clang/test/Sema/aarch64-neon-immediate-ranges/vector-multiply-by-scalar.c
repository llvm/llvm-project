// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_vector_multiply_by_scalar_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vmul_lane_s16(arg_i16x4, arg_i16x4, 0);
	vmul_lane_s16(arg_i16x4, arg_i16x4, 3);
	vmul_lane_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_s16(arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_s16(arg_i16x8, arg_i16x4, 0);
	vmulq_lane_s16(arg_i16x8, arg_i16x4, 3);
	vmulq_lane_s16(arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_s16(arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_s16(arg_i16x4, arg_i16x8, 0);
	vmul_laneq_s16(arg_i16x4, arg_i16x8, 7);
	vmul_laneq_s16(arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_s16(arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_s16(arg_i16x8, arg_i16x8, 0);
	vmulq_laneq_s16(arg_i16x8, arg_i16x8, 7);
	vmulq_laneq_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_s16(arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vmul_lane_s32(arg_i32x2, arg_i32x2, 0);
	vmul_lane_s32(arg_i32x2, arg_i32x2, 1);
	vmul_lane_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_s32(arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_s32(arg_i32x4, arg_i32x2, 0);
	vmulq_lane_s32(arg_i32x4, arg_i32x2, 1);
	vmulq_lane_s32(arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_s32(arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_s32(arg_i32x2, arg_i32x4, 0);
	vmul_laneq_s32(arg_i32x2, arg_i32x4, 3);
	vmul_laneq_s32(arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_s32(arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_s32(arg_i32x4, arg_i32x4, 0);
	vmulq_laneq_s32(arg_i32x4, arg_i32x4, 3);
	vmulq_laneq_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_s32(arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vmul_lane_u16(arg_u16x4, arg_u16x4, 0);
	vmul_lane_u16(arg_u16x4, arg_u16x4, 3);
	vmul_lane_u16(arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_u16(arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_u16(arg_u16x8, arg_u16x4, 0);
	vmulq_lane_u16(arg_u16x8, arg_u16x4, 3);
	vmulq_lane_u16(arg_u16x8, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_u16(arg_u16x8, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_u16(arg_u16x4, arg_u16x8, 0);
	vmul_laneq_u16(arg_u16x4, arg_u16x8, 7);
	vmul_laneq_u16(arg_u16x4, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_u16(arg_u16x4, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_u16(arg_u16x8, arg_u16x8, 0);
	vmulq_laneq_u16(arg_u16x8, arg_u16x8, 7);
	vmulq_laneq_u16(arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_u16(arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vmul_lane_u32(arg_u32x2, arg_u32x2, 0);
	vmul_lane_u32(arg_u32x2, arg_u32x2, 1);
	vmul_lane_u32(arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_u32(arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_u32(arg_u32x4, arg_u32x2, 0);
	vmulq_lane_u32(arg_u32x4, arg_u32x2, 1);
	vmulq_lane_u32(arg_u32x4, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_u32(arg_u32x4, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_u32(arg_u32x2, arg_u32x4, 0);
	vmul_laneq_u32(arg_u32x2, arg_u32x4, 3);
	vmul_laneq_u32(arg_u32x2, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_u32(arg_u32x2, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_u32(arg_u32x4, arg_u32x4, 0);
	vmulq_laneq_u32(arg_u32x4, arg_u32x4, 3);
	vmulq_laneq_u32(arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_u32(arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_f32(float32_t arg_f32, float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vmul_lane_f32(arg_f32x2, arg_f32x2, 0);
	vmul_lane_f32(arg_f32x2, arg_f32x2, 1);
	vmul_lane_f32(arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_f32(arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_f32(arg_f32x4, arg_f32x2, 0);
	vmulq_lane_f32(arg_f32x4, arg_f32x2, 1);
	vmulq_lane_f32(arg_f32x4, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_f32(arg_f32x4, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmuls_lane_f32(arg_f32, arg_f32x2, 0);
	vmuls_lane_f32(arg_f32, arg_f32x2, 1);
	vmuls_lane_f32(arg_f32, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmuls_lane_f32(arg_f32, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_f32(arg_f32x2, arg_f32x4, 0);
	vmul_laneq_f32(arg_f32x2, arg_f32x4, 3);
	vmul_laneq_f32(arg_f32x2, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_f32(arg_f32x2, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_f32(arg_f32x4, arg_f32x4, 0);
	vmulq_laneq_f32(arg_f32x4, arg_f32x4, 3);
	vmulq_laneq_f32(arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_f32(arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmuls_laneq_f32(arg_f32, arg_f32x4, 0);
	vmuls_laneq_f32(arg_f32, arg_f32x4, 3);
	vmuls_laneq_f32(arg_f32, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmuls_laneq_f32(arg_f32, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_f64(float64x1_t arg_f64x1, float64_t arg_f64, float64x2_t arg_f64x2) {
	vmul_lane_f64(arg_f64x1, arg_f64x1, 0);
	vmul_lane_f64(arg_f64x1, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_f64(arg_f64x1, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_f64(arg_f64x2, arg_f64x1, 0);
	vmulq_lane_f64(arg_f64x2, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_f64(arg_f64x2, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmuld_lane_f64(arg_f64, arg_f64x1, 0);
	vmuld_lane_f64(arg_f64, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmuld_lane_f64(arg_f64, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_f64(arg_f64x1, arg_f64x2, 0);
	vmul_laneq_f64(arg_f64x1, arg_f64x2, 1);
	vmul_laneq_f64(arg_f64x1, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_f64(arg_f64x1, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_f64(arg_f64x2, arg_f64x2, 0);
	vmulq_laneq_f64(arg_f64x2, arg_f64x2, 1);
	vmulq_laneq_f64(arg_f64x2, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_f64(arg_f64x2, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmuld_laneq_f64(arg_f64, arg_f64x2, 0);
	vmuld_laneq_f64(arg_f64, arg_f64x2, 1);
	vmuld_laneq_f64(arg_f64, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmuld_laneq_f64(arg_f64, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}
