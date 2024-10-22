// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_extract_one_element_from_vector_s8(int8x16_t arg_i8x16, int8x8_t arg_i8x8) {
	vdupb_lane_s8(arg_i8x8, 0);
	vdupb_lane_s8(arg_i8x8, 7);
	vdupb_lane_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupb_lane_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupb_laneq_s8(arg_i8x16, 0);
	vdupb_laneq_s8(arg_i8x16, 15);
	vdupb_laneq_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupb_laneq_s8(arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_s8(arg_i8x8, 0);
	vget_lane_s8(arg_i8x8, 7);
	vget_lane_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_s8(arg_i8x16, 0);
	vgetq_lane_s8(arg_i8x16, 15);
	vgetq_lane_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_s8(arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vduph_lane_s16(arg_i16x4, 0);
	vduph_lane_s16(arg_i16x4, 3);
	vduph_lane_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_lane_s16(arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vduph_laneq_s16(arg_i16x8, 0);
	vduph_laneq_s16(arg_i16x8, 7);
	vduph_laneq_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_laneq_s16(arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_s16(arg_i16x4, 0);
	vget_lane_s16(arg_i16x4, 3);
	vget_lane_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_s16(arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_s16(arg_i16x8, 0);
	vgetq_lane_s16(arg_i16x8, 7);
	vgetq_lane_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_s16(arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_s32(int32x4_t arg_i32x4, int32x2_t arg_i32x2) {
	vdups_lane_s32(arg_i32x2, 0);
	vdups_lane_s32(arg_i32x2, 1);
	vdups_lane_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdups_lane_s32(arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdups_laneq_s32(arg_i32x4, 0);
	vdups_laneq_s32(arg_i32x4, 3);
	vdups_laneq_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdups_laneq_s32(arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_s32(arg_i32x2, 0);
	vget_lane_s32(arg_i32x2, 1);
	vget_lane_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_s32(arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_s32(arg_i32x4, 0);
	vgetq_lane_s32(arg_i32x4, 3);
	vgetq_lane_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_s32(arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_s64(int64x2_t arg_i64x2, int64x1_t arg_i64x1) {
	vdupd_lane_s64(arg_i64x1, 0);
	vdupd_lane_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupd_lane_s64(arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupd_laneq_s64(arg_i64x2, 0);
	vdupd_laneq_s64(arg_i64x2, 1);
	vdupd_laneq_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupd_laneq_s64(arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_s64(arg_i64x1, 0);
	vget_lane_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_s64(arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_s64(arg_i64x2, 0);
	vgetq_lane_s64(arg_i64x2, 1);
	vgetq_lane_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_s64(arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vdupb_lane_u8(arg_u8x8, 0);
	vdupb_lane_u8(arg_u8x8, 7);
	vdupb_lane_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupb_lane_u8(arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupb_laneq_u8(arg_u8x16, 0);
	vdupb_laneq_u8(arg_u8x16, 15);
	vdupb_laneq_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupb_laneq_u8(arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_u8(arg_u8x8, 0);
	vget_lane_u8(arg_u8x8, 7);
	vget_lane_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_u8(arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_u8(arg_u8x16, 0);
	vgetq_lane_u8(arg_u8x16, 15);
	vgetq_lane_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_u8(arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vduph_lane_u16(arg_u16x4, 0);
	vduph_lane_u16(arg_u16x4, 3);
	vduph_lane_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_lane_u16(arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vduph_laneq_u16(arg_u16x8, 0);
	vduph_laneq_u16(arg_u16x8, 7);
	vduph_laneq_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_laneq_u16(arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_u16(arg_u16x4, 0);
	vget_lane_u16(arg_u16x4, 3);
	vget_lane_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_u16(arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_u16(arg_u16x8, 0);
	vgetq_lane_u16(arg_u16x8, 7);
	vgetq_lane_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_u16(arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vdups_lane_u32(arg_u32x2, 0);
	vdups_lane_u32(arg_u32x2, 1);
	vdups_lane_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdups_lane_u32(arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdups_laneq_u32(arg_u32x4, 0);
	vdups_laneq_u32(arg_u32x4, 3);
	vdups_laneq_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdups_laneq_u32(arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_u32(arg_u32x2, 0);
	vget_lane_u32(arg_u32x2, 1);
	vget_lane_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_u32(arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_u32(arg_u32x4, 0);
	vgetq_lane_u32(arg_u32x4, 3);
	vgetq_lane_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_u32(arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_u64(uint64x1_t arg_u64x1, uint64x2_t arg_u64x2) {
	vdupd_lane_u64(arg_u64x1, 0);
	vdupd_lane_u64(arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupd_lane_u64(arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupd_laneq_u64(arg_u64x2, 0);
	vdupd_laneq_u64(arg_u64x2, 1);
	vdupd_laneq_u64(arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupd_laneq_u64(arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_u64(arg_u64x1, 0);
	vget_lane_u64(arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_u64(arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_u64(arg_u64x2, 0);
	vgetq_lane_u64(arg_u64x2, 1);
	vgetq_lane_u64(arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_u64(arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_f32(float32x4_t arg_f32x4, float32x2_t arg_f32x2) {
	vdups_lane_f32(arg_f32x2, 0);
	vdups_lane_f32(arg_f32x2, 1);
	vdups_lane_f32(arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdups_lane_f32(arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdups_laneq_f32(arg_f32x4, 0);
	vdups_laneq_f32(arg_f32x4, 3);
	vdups_laneq_f32(arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdups_laneq_f32(arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_f32(arg_f32x2, 0);
	vget_lane_f32(arg_f32x2, 1);
	vget_lane_f32(arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_f32(arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_f32(arg_f32x4, 0);
	vgetq_lane_f32(arg_f32x4, 3);
	vgetq_lane_f32(arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_f32(arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_f64(float64x2_t arg_f64x2, float64x1_t arg_f64x1) {
	vdupd_lane_f64(arg_f64x1, 0);
	vdupd_lane_f64(arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupd_lane_f64(arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupd_laneq_f64(arg_f64x2, 0);
	vdupd_laneq_f64(arg_f64x2, 1);
	vdupd_laneq_f64(arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupd_laneq_f64(arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_f64(arg_f64x1, 0);
	vget_lane_f64(arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_f64(arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_f64(arg_f64x2, 0);
	vgetq_lane_f64(arg_f64x2, 1);
	vgetq_lane_f64(arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_f64(arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_p8(poly8x16_t arg_p8x16, poly8x8_t arg_p8x8) {
	vdupb_lane_p8(arg_p8x8, 0);
	vdupb_lane_p8(arg_p8x8, 7);
	vdupb_lane_p8(arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupb_lane_p8(arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupb_laneq_p8(arg_p8x16, 0);
	vdupb_laneq_p8(arg_p8x16, 15);
	vdupb_laneq_p8(arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupb_laneq_p8(arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_p8(arg_p8x8, 0);
	vget_lane_p8(arg_p8x8, 7);
	vget_lane_p8(arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_p8(arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_p8(arg_p8x16, 0);
	vgetq_lane_p8(arg_p8x16, 15);
	vgetq_lane_p8(arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_p8(arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_p16(poly16x4_t arg_p16x4, poly16x8_t arg_p16x8) {
	vduph_lane_p16(arg_p16x4, 0);
	vduph_lane_p16(arg_p16x4, 3);
	vduph_lane_p16(arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_lane_p16(arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vduph_laneq_p16(arg_p16x8, 0);
	vduph_laneq_p16(arg_p16x8, 7);
	vduph_laneq_p16(arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_laneq_p16(arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vget_lane_p16(arg_p16x4, 0);
	vget_lane_p16(arg_p16x4, 3);
	vget_lane_p16(arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_p16(arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_p16(arg_p16x8, 0);
	vgetq_lane_p16(arg_p16x8, 7);
	vgetq_lane_p16(arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_p16(arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_p64(poly64x2_t arg_p64x2, poly64x1_t arg_p64x1) {
	vget_lane_p64(arg_p64x1, 0);
	vget_lane_p64(arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_p64(arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_p64(arg_p64x2, 0);
	vgetq_lane_p64(arg_p64x2, 1);
	vgetq_lane_p64(arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_p64(arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_one_element_from_vector_f16(float16x8_t arg_f16x8, float16x4_t arg_f16x4) {
	vget_lane_f16(arg_f16x4, 0);
	vget_lane_f16(arg_f16x4, 3);
	vget_lane_f16(arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_f16(arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_f16(arg_f16x8, 0);
	vgetq_lane_f16(arg_f16x8, 7);
	vgetq_lane_f16(arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_f16(arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

