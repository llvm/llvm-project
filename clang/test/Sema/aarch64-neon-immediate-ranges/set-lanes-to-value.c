// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_set_all_lanes_to_the_same_value_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vdup_lane_s8(arg_i8x8, 0);
	vdup_lane_s8(arg_i8x8, 7);
	vdup_lane_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_s8(arg_i8x8, 0);
	vdupq_lane_s8(arg_i8x8, 7);
	vdupq_lane_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_s8(arg_i8x16, 0);
	vdup_laneq_s8(arg_i8x16, 15);
	vdup_laneq_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_s8(arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_s8(arg_i8x16, 0);
	vdupq_laneq_s8(arg_i8x16, 15);
	vdupq_laneq_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_s8(arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vdup_lane_s16(arg_i16x4, 0);
	vdup_lane_s16(arg_i16x4, 3);
	vdup_lane_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_s16(arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_s16(arg_i16x4, 0);
	vdupq_lane_s16(arg_i16x4, 3);
	vdupq_lane_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_s16(arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_s16(arg_i16x8, 0);
	vdup_laneq_s16(arg_i16x8, 7);
	vdup_laneq_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_s16(arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_s16(arg_i16x8, 0);
	vdupq_laneq_s16(arg_i16x8, 7);
	vdupq_laneq_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_s16(arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_s32(int32x4_t arg_i32x4, int32x2_t arg_i32x2) {
	vdup_lane_s32(arg_i32x2, 0);
	vdup_lane_s32(arg_i32x2, 1);
	vdup_lane_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_s32(arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_s32(arg_i32x2, 0);
	vdupq_lane_s32(arg_i32x2, 1);
	vdupq_lane_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_s32(arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_s32(arg_i32x4, 0);
	vdup_laneq_s32(arg_i32x4, 3);
	vdup_laneq_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_s32(arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_s32(arg_i32x4, 0);
	vdupq_laneq_s32(arg_i32x4, 3);
	vdupq_laneq_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_s32(arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_s64(int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vdup_lane_s64(arg_i64x1, 0);
	vdup_lane_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_s64(arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_s64(arg_i64x1, 0);
	vdupq_lane_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_s64(arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_s64(arg_i64x2, 0);
	vdup_laneq_s64(arg_i64x2, 1);
	vdup_laneq_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_s64(arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_s64(arg_i64x2, 0);
	vdupq_laneq_s64(arg_i64x2, 1);
	vdupq_laneq_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_s64(arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vdup_lane_u8(arg_u8x8, 0);
	vdup_lane_u8(arg_u8x8, 7);
	vdup_lane_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_u8(arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_u8(arg_u8x8, 0);
	vdupq_lane_u8(arg_u8x8, 7);
	vdupq_lane_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_u8(arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_u8(arg_u8x16, 0);
	vdup_laneq_u8(arg_u8x16, 15);
	vdup_laneq_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_u8(arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_u8(arg_u8x16, 0);
	vdupq_laneq_u8(arg_u8x16, 15);
	vdupq_laneq_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_u8(arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vdup_lane_u16(arg_u16x4, 0);
	vdup_lane_u16(arg_u16x4, 3);
	vdup_lane_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_u16(arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_u16(arg_u16x4, 0);
	vdupq_lane_u16(arg_u16x4, 3);
	vdupq_lane_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_u16(arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_u16(arg_u16x8, 0);
	vdup_laneq_u16(arg_u16x8, 7);
	vdup_laneq_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_u16(arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_u16(arg_u16x8, 0);
	vdupq_laneq_u16(arg_u16x8, 7);
	vdupq_laneq_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_u16(arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_u32(uint32x4_t arg_u32x4, uint32x2_t arg_u32x2) {
	vdup_lane_u32(arg_u32x2, 0);
	vdup_lane_u32(arg_u32x2, 1);
	vdup_lane_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_u32(arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_u32(arg_u32x2, 0);
	vdupq_lane_u32(arg_u32x2, 1);
	vdupq_lane_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_u32(arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_u32(arg_u32x4, 0);
	vdup_laneq_u32(arg_u32x4, 3);
	vdup_laneq_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_u32(arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_u32(arg_u32x4, 0);
	vdupq_laneq_u32(arg_u32x4, 3);
	vdupq_laneq_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_u32(arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_u64(uint64x1_t arg_u64x1, uint64x2_t arg_u64x2) {
	vdup_lane_u64(arg_u64x1, 0);
	vdup_lane_u64(arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_u64(arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_u64(arg_u64x1, 0);
	vdupq_lane_u64(arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_u64(arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_u64(arg_u64x2, 0);
	vdup_laneq_u64(arg_u64x2, 1);
	vdup_laneq_u64(arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_u64(arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_u64(arg_u64x2, 0);
	vdupq_laneq_u64(arg_u64x2, 1);
	vdupq_laneq_u64(arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_u64(arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_p64(poly64x1_t arg_p64x1, poly64x2_t arg_p64x2) {
	vdup_lane_p64(arg_p64x1, 0);
	vdup_lane_p64(arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_p64(arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_p64(arg_p64x1, 0);
	vdupq_lane_p64(arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_p64(arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_p64(arg_p64x2, 0);
	vdup_laneq_p64(arg_p64x2, 1);
	vdup_laneq_p64(arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_p64(arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_p64(arg_p64x2, 0);
	vdupq_laneq_p64(arg_p64x2, 1);
	vdupq_laneq_p64(arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_p64(arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_f32(float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vdup_lane_f32(arg_f32x2, 0);
	vdup_lane_f32(arg_f32x2, 1);
	vdup_lane_f32(arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_f32(arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_f32(arg_f32x2, 0);
	vdupq_lane_f32(arg_f32x2, 1);
	vdupq_lane_f32(arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_f32(arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_f32(arg_f32x4, 0);
	vdup_laneq_f32(arg_f32x4, 3);
	vdup_laneq_f32(arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_f32(arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_f32(arg_f32x4, 0);
	vdupq_laneq_f32(arg_f32x4, 3);
	vdupq_laneq_f32(arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_f32(arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_p8(poly8x16_t arg_p8x16, poly8x8_t arg_p8x8) {
	vdup_lane_p8(arg_p8x8, 0);
	vdup_lane_p8(arg_p8x8, 7);
	vdup_lane_p8(arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_p8(arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_p8(arg_p8x8, 0);
	vdupq_lane_p8(arg_p8x8, 7);
	vdupq_lane_p8(arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_p8(arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_p8(arg_p8x16, 0);
	vdup_laneq_p8(arg_p8x16, 15);
	vdup_laneq_p8(arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_p8(arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_p8(arg_p8x16, 0);
	vdupq_laneq_p8(arg_p8x16, 15);
	vdupq_laneq_p8(arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_p8(arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_p16(poly16x4_t arg_p16x4, poly16x8_t arg_p16x8) {
	vdup_lane_p16(arg_p16x4, 0);
	vdup_lane_p16(arg_p16x4, 3);
	vdup_lane_p16(arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_p16(arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_p16(arg_p16x4, 0);
	vdupq_lane_p16(arg_p16x4, 3);
	vdupq_lane_p16(arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_p16(arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_p16(arg_p16x8, 0);
	vdup_laneq_p16(arg_p16x8, 7);
	vdup_laneq_p16(arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_p16(arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_p16(arg_p16x8, 0);
	vdupq_laneq_p16(arg_p16x8, 7);
	vdupq_laneq_p16(arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_p16(arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_all_lanes_to_the_same_value_f64(float64x2_t arg_f64x2, float64x1_t arg_f64x1) {
	vdup_lane_f64(arg_f64x1, 0);
	vdup_lane_f64(arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_f64(arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_f64(arg_f64x1, 0);
	vdupq_lane_f64(arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_f64(arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_f64(arg_f64x2, 0);
	vdup_laneq_f64(arg_f64x2, 1);
	vdup_laneq_f64(arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_f64(arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_f64(arg_f64x2, 0);
	vdupq_laneq_f64(arg_f64x2, 1);
	vdupq_laneq_f64(arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_f64(arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

