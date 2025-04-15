// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_copy_vector_lane_s8(int8x16_t arg_i8x16, int8x8_t arg_i8x8) {
	vcopy_lane_s8(arg_i8x8, 0, arg_i8x8, 0);

	vcopy_lane_s8(arg_i8x8, 7, arg_i8x8, 0);
	vcopy_lane_s8(arg_i8x8, -1, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s8(arg_i8x8, 8, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_s8(arg_i8x8, 0, arg_i8x8, 7);
	vcopy_lane_s8(arg_i8x8, 0, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s8(arg_i8x8, 0, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s8(arg_i8x16, 0, arg_i8x8, 0);
	vcopyq_lane_s8(arg_i8x16, 15, arg_i8x8, 0);
	vcopyq_lane_s8(arg_i8x16, -1, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s8(arg_i8x16, 16, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s8(arg_i8x16, 0, arg_i8x8, 7);
	vcopyq_lane_s8(arg_i8x16, 0, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s8(arg_i8x16, 0, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s8(arg_i8x8, 0, arg_i8x16, 0);
	vcopy_laneq_s8(arg_i8x8, 7, arg_i8x16, 0);
	vcopy_laneq_s8(arg_i8x8, -1, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s8(arg_i8x8, 8, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s8(arg_i8x8, 0, arg_i8x16, 15);
	vcopy_laneq_s8(arg_i8x8, 0, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s8(arg_i8x8, 0, arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s8(arg_i8x16, 0, arg_i8x16, 0);
	vcopyq_laneq_s8(arg_i8x16, 15, arg_i8x16, 0);
	vcopyq_laneq_s8(arg_i8x16, -1, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s8(arg_i8x16, 16, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s8(arg_i8x16, 0, arg_i8x16, 15);
	vcopyq_laneq_s8(arg_i8x16, 0, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s8(arg_i8x16, 0, arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vcopy_lane_s16(arg_i16x4, 0, arg_i16x4, 0);
	vcopy_lane_s16(arg_i16x4, 3, arg_i16x4, 0);
	vcopy_lane_s16(arg_i16x4, -1, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s16(arg_i16x4, 4, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_s16(arg_i16x4, 0, arg_i16x4, 3);
	vcopy_lane_s16(arg_i16x4, 0, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s16(arg_i16x4, 0, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s16(arg_i16x8, 0, arg_i16x4, 0);
	vcopyq_lane_s16(arg_i16x8, 7, arg_i16x4, 0);
	vcopyq_lane_s16(arg_i16x8, -1, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s16(arg_i16x8, 8, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s16(arg_i16x8, 0, arg_i16x4, 3);
	vcopyq_lane_s16(arg_i16x8, 0, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s16(arg_i16x8, 0, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s16(arg_i16x4, 0, arg_i16x8, 0);
	vcopy_laneq_s16(arg_i16x4, 3, arg_i16x8, 0);
	vcopy_laneq_s16(arg_i16x4, -1, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s16(arg_i16x4, 4, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s16(arg_i16x4, 0, arg_i16x8, 7);
	vcopy_laneq_s16(arg_i16x4, 0, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s16(arg_i16x4, 0, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s16(arg_i16x8, 0, arg_i16x8, 0);
	vcopyq_laneq_s16(arg_i16x8, 7, arg_i16x8, 0);
	vcopyq_laneq_s16(arg_i16x8, -1, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s16(arg_i16x8, 8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s16(arg_i16x8, 0, arg_i16x8, 7);
	vcopyq_laneq_s16(arg_i16x8, 0, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s16(arg_i16x8, 0, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vcopy_lane_s32(arg_i32x2, 0, arg_i32x2, 0);
	vcopy_lane_s32(arg_i32x2, 1, arg_i32x2, 0);
	vcopy_lane_s32(arg_i32x2, -1, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s32(arg_i32x2, 2, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_s32(arg_i32x2, 0, arg_i32x2, 1);
	vcopy_lane_s32(arg_i32x2, 0, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s32(arg_i32x2, 0, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s32(arg_i32x4, 0, arg_i32x2, 0);
	vcopyq_lane_s32(arg_i32x4, 3, arg_i32x2, 0);
	vcopyq_lane_s32(arg_i32x4, -1, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s32(arg_i32x4, 4, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s32(arg_i32x4, 0, arg_i32x2, 1);
	vcopyq_lane_s32(arg_i32x4, 0, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s32(arg_i32x4, 0, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s32(arg_i32x2, 0, arg_i32x4, 0);
	vcopy_laneq_s32(arg_i32x2, 1, arg_i32x4, 0);
	vcopy_laneq_s32(arg_i32x2, -1, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s32(arg_i32x2, 2, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s32(arg_i32x2, 0, arg_i32x4, 3);
	vcopy_laneq_s32(arg_i32x2, 0, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s32(arg_i32x2, 0, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s32(arg_i32x4, 0, arg_i32x4, 0);
	vcopyq_laneq_s32(arg_i32x4, 3, arg_i32x4, 0);
	vcopyq_laneq_s32(arg_i32x4, -1, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s32(arg_i32x4, 4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s32(arg_i32x4, 0, arg_i32x4, 3);
	vcopyq_laneq_s32(arg_i32x4, 0, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s32(arg_i32x4, 0, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_s64(int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vcopy_lane_s64(arg_i64x1, 0, arg_i64x1, 0);
	vcopy_lane_s64(arg_i64x1, -1, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s64(arg_i64x1, 1, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s64(arg_i64x1, 0, arg_i64x1, 0);
	vcopy_lane_s64(arg_i64x1, 0, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_s64(arg_i64x1, 0, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s64(arg_i64x2, 0, arg_i64x1, 0);
	vcopyq_lane_s64(arg_i64x2, 1, arg_i64x1, 0);
	vcopyq_lane_s64(arg_i64x2, -1, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s64(arg_i64x2, 2, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_s64(arg_i64x2, 0, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_s64(arg_i64x2, 0, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s64(arg_i64x1, 0, arg_i64x2, 0);
	vcopy_laneq_s64(arg_i64x1, -1, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s64(arg_i64x1, 1, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_s64(arg_i64x1, 0, arg_i64x2, 1);
	vcopy_laneq_s64(arg_i64x1, 0, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_s64(arg_i64x1, 0, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s64(arg_i64x2, 0, arg_i64x2, 0);
	vcopyq_laneq_s64(arg_i64x2, 1, arg_i64x2, 0);
	vcopyq_laneq_s64(arg_i64x2, -1, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s64(arg_i64x2, 2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_s64(arg_i64x2, 0, arg_i64x2, 1);
	vcopyq_laneq_s64(arg_i64x2, 0, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_s64(arg_i64x2, 0, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vcopy_lane_u8(arg_u8x8, 0, arg_u8x8, 0);
	vcopy_lane_u8(arg_u8x8, 7, arg_u8x8, 0);
	vcopy_lane_u8(arg_u8x8, -1, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u8(arg_u8x8, 8, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_u8(arg_u8x8, 0, arg_u8x8, 7);
	vcopy_lane_u8(arg_u8x8, 0, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u8(arg_u8x8, 0, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u8(arg_u8x16, 0, arg_u8x8, 0);
	vcopyq_lane_u8(arg_u8x16, 15, arg_u8x8, 0);
	vcopyq_lane_u8(arg_u8x16, -1, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u8(arg_u8x16, 16, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u8(arg_u8x16, 0, arg_u8x8, 7);
	vcopyq_lane_u8(arg_u8x16, 0, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u8(arg_u8x16, 0, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u8(arg_u8x8, 0, arg_u8x16, 0);
	vcopy_laneq_u8(arg_u8x8, 7, arg_u8x16, 0);
	vcopy_laneq_u8(arg_u8x8, -1, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u8(arg_u8x8, 8, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u8(arg_u8x8, 0, arg_u8x16, 15);
	vcopy_laneq_u8(arg_u8x8, 0, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u8(arg_u8x8, 0, arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u8(arg_u8x16, 0, arg_u8x16, 0);
	vcopyq_laneq_u8(arg_u8x16, 15, arg_u8x16, 0);
	vcopyq_laneq_u8(arg_u8x16, -1, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u8(arg_u8x16, 16, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u8(arg_u8x16, 0, arg_u8x16, 15);
	vcopyq_laneq_u8(arg_u8x16, 0, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u8(arg_u8x16, 0, arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vcopy_lane_u16(arg_u16x4, 0, arg_u16x4, 0);
	vcopy_lane_u16(arg_u16x4, 3, arg_u16x4, 0);
	vcopy_lane_u16(arg_u16x4, -1, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u16(arg_u16x4, 4, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_u16(arg_u16x4, 0, arg_u16x4, 3);
	vcopy_lane_u16(arg_u16x4, 0, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u16(arg_u16x4, 0, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u16(arg_u16x8, 0, arg_u16x4, 0);
	vcopyq_lane_u16(arg_u16x8, 7, arg_u16x4, 0);
	vcopyq_lane_u16(arg_u16x8, -1, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u16(arg_u16x8, 8, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u16(arg_u16x8, 0, arg_u16x4, 3);
	vcopyq_lane_u16(arg_u16x8, 0, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u16(arg_u16x8, 0, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u16(arg_u16x4, 0, arg_u16x8, 0);
	vcopy_laneq_u16(arg_u16x4, 3, arg_u16x8, 0);
	vcopy_laneq_u16(arg_u16x4, -1, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u16(arg_u16x4, 4, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u16(arg_u16x4, 0, arg_u16x8, 7);
	vcopy_laneq_u16(arg_u16x4, 0, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u16(arg_u16x4, 0, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u16(arg_u16x8, 0, arg_u16x8, 0);
	vcopyq_laneq_u16(arg_u16x8, 7, arg_u16x8, 0);
	vcopyq_laneq_u16(arg_u16x8, -1, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u16(arg_u16x8, 8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u16(arg_u16x8, 0, arg_u16x8, 7);
	vcopyq_laneq_u16(arg_u16x8, 0, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u16(arg_u16x8, 0, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vcopy_lane_u32(arg_u32x2, 0, arg_u32x2, 0);
	vcopy_lane_u32(arg_u32x2, 1, arg_u32x2, 0);
	vcopy_lane_u32(arg_u32x2, -1, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u32(arg_u32x2, 2, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_u32(arg_u32x2, 0, arg_u32x2, 1);
	vcopy_lane_u32(arg_u32x2, 0, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u32(arg_u32x2, 0, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u32(arg_u32x4, 0, arg_u32x2, 0);
	vcopyq_lane_u32(arg_u32x4, 3, arg_u32x2, 0);
	vcopyq_lane_u32(arg_u32x4, -1, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u32(arg_u32x4, 4, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u32(arg_u32x4, 0, arg_u32x2, 1);
	vcopyq_lane_u32(arg_u32x4, 0, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u32(arg_u32x4, 0, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u32(arg_u32x2, 0, arg_u32x4, 0);
	vcopy_laneq_u32(arg_u32x2, 1, arg_u32x4, 0);
	vcopy_laneq_u32(arg_u32x2, -1, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u32(arg_u32x2, 2, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u32(arg_u32x2, 0, arg_u32x4, 3);
	vcopy_laneq_u32(arg_u32x2, 0, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u32(arg_u32x2, 0, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u32(arg_u32x4, 0, arg_u32x4, 0);
	vcopyq_laneq_u32(arg_u32x4, 3, arg_u32x4, 0);
	vcopyq_laneq_u32(arg_u32x4, -1, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u32(arg_u32x4, 4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u32(arg_u32x4, 0, arg_u32x4, 3);
	vcopyq_laneq_u32(arg_u32x4, 0, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u32(arg_u32x4, 0, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_u64(uint64x2_t arg_u64x2, uint64x1_t arg_u64x1) {
	vcopy_lane_u64(arg_u64x1, 0, arg_u64x1, 0);
	vcopy_lane_u64(arg_u64x1, -1, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u64(arg_u64x1, 1, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_u64(arg_u64x1, 0, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_u64(arg_u64x1, 0, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u64(arg_u64x2, 0, arg_u64x1, 0);
	vcopyq_lane_u64(arg_u64x2, 1, arg_u64x1, 0);
	vcopyq_lane_u64(arg_u64x2, -1, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u64(arg_u64x2, 2, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_u64(arg_u64x2, 0, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_u64(arg_u64x2, 0, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u64(arg_u64x1, 0, arg_u64x2, 0);
	vcopy_laneq_u64(arg_u64x1, -1, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u64(arg_u64x1, 1, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_u64(arg_u64x1, 0, arg_u64x2, 1);
	vcopy_laneq_u64(arg_u64x1, 0, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_u64(arg_u64x1, 0, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u64(arg_u64x2, 0, arg_u64x2, 0);
	vcopyq_laneq_u64(arg_u64x2, 1, arg_u64x2, 0);
	vcopyq_laneq_u64(arg_u64x2, -1, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u64(arg_u64x2, 2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_u64(arg_u64x2, 0, arg_u64x2, 1);
	vcopyq_laneq_u64(arg_u64x2, 0, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_u64(arg_u64x2, 0, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_p64(poly64x1_t arg_p64x1, poly64x2_t arg_p64x2) {
	vcopy_lane_p64(arg_p64x1, 0, arg_p64x1, 0);
	vcopy_lane_p64(arg_p64x1, -1, arg_p64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_p64(arg_p64x1, 1, arg_p64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_p64(arg_p64x1, 0, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_p64(arg_p64x1, 0, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_p64(arg_p64x2, 0, arg_p64x1, 0);
	vcopyq_lane_p64(arg_p64x2, 1, arg_p64x1, 0);
	vcopyq_lane_p64(arg_p64x2, -1, arg_p64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_p64(arg_p64x2, 2, arg_p64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_p64(arg_p64x2, 0, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_p64(arg_p64x2, 0, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_p64(arg_p64x1, 0, arg_p64x2, 0);
	vcopy_laneq_p64(arg_p64x1, -1, arg_p64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_p64(arg_p64x1, 1, arg_p64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_p64(arg_p64x1, 0, arg_p64x2, 1);
	vcopy_laneq_p64(arg_p64x1, 0, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_p64(arg_p64x1, 0, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_p64(arg_p64x2, 0, arg_p64x2, 0);
	vcopyq_laneq_p64(arg_p64x2, 1, arg_p64x2, 0);
	vcopyq_laneq_p64(arg_p64x2, -1, arg_p64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_p64(arg_p64x2, 2, arg_p64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_p64(arg_p64x2, 0, arg_p64x2, 1);
	vcopyq_laneq_p64(arg_p64x2, 0, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_p64(arg_p64x2, 0, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_f32(float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vcopy_lane_f32(arg_f32x2, 0, arg_f32x2, 0);
	vcopy_lane_f32(arg_f32x2, 1, arg_f32x2, 0);
	vcopy_lane_f32(arg_f32x2, -1, arg_f32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_f32(arg_f32x2, 2, arg_f32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_f32(arg_f32x2, 0, arg_f32x2, 1);
	vcopy_lane_f32(arg_f32x2, 0, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_f32(arg_f32x2, 0, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_f32(arg_f32x4, 0, arg_f32x2, 0);
	vcopyq_lane_f32(arg_f32x4, 3, arg_f32x2, 0);
	vcopyq_lane_f32(arg_f32x4, -1, arg_f32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_f32(arg_f32x4, 4, arg_f32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_f32(arg_f32x4, 0, arg_f32x2, 1);
	vcopyq_lane_f32(arg_f32x4, 0, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_f32(arg_f32x4, 0, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_f32(arg_f32x2, 0, arg_f32x4, 0);
	vcopy_laneq_f32(arg_f32x2, 1, arg_f32x4, 0);
	vcopy_laneq_f32(arg_f32x2, -1, arg_f32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_f32(arg_f32x2, 2, arg_f32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_f32(arg_f32x2, 0, arg_f32x4, 3);
	vcopy_laneq_f32(arg_f32x2, 0, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_f32(arg_f32x2, 0, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_f32(arg_f32x4, 0, arg_f32x4, 0);
	vcopyq_laneq_f32(arg_f32x4, 3, arg_f32x4, 0);
	vcopyq_laneq_f32(arg_f32x4, -1, arg_f32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_f32(arg_f32x4, 4, arg_f32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_f32(arg_f32x4, 0, arg_f32x4, 3);
	vcopyq_laneq_f32(arg_f32x4, 0, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_f32(arg_f32x4, 0, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_f64(float64x2_t arg_f64x2, float64x1_t arg_f64x1) {
	vcopy_lane_f64(arg_f64x1, 0, arg_f64x1, 0);
	vcopy_lane_f64(arg_f64x1, -1, arg_f64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_f64(arg_f64x1, 1, arg_f64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_f64(arg_f64x1, 0, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_f64(arg_f64x1, 0, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_f64(arg_f64x2, 0, arg_f64x1, 0);
	vcopyq_lane_f64(arg_f64x2, 1, arg_f64x1, 0);
	vcopyq_lane_f64(arg_f64x2, -1, arg_f64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_f64(arg_f64x2, 2, arg_f64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_f64(arg_f64x2, 0, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_f64(arg_f64x2, 0, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_f64(arg_f64x1, 0, arg_f64x2, 0);
	vcopy_laneq_f64(arg_f64x1, -1, arg_f64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_f64(arg_f64x1, 1, arg_f64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_f64(arg_f64x1, 0, arg_f64x2, 1);
	vcopy_laneq_f64(arg_f64x1, 0, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_f64(arg_f64x1, 0, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_f64(arg_f64x2, 0, arg_f64x2, 0);
	vcopyq_laneq_f64(arg_f64x2, 1, arg_f64x2, 0);
	vcopyq_laneq_f64(arg_f64x2, -1, arg_f64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_f64(arg_f64x2, 2, arg_f64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_f64(arg_f64x2, 0, arg_f64x2, 1);
	vcopyq_laneq_f64(arg_f64x2, 0, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_f64(arg_f64x2, 0, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_p8(poly8x16_t arg_p8x16, poly8x8_t arg_p8x8) {
	vcopy_lane_p8(arg_p8x8, 0, arg_p8x8, 0);
	vcopy_lane_p8(arg_p8x8, 7, arg_p8x8, 0);
	vcopy_lane_p8(arg_p8x8, -1, arg_p8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_p8(arg_p8x8, 8, arg_p8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_p8(arg_p8x8, 0, arg_p8x8, 7);
	vcopy_lane_p8(arg_p8x8, 0, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_p8(arg_p8x8, 0, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_p8(arg_p8x16, 0, arg_p8x8, 0);
	vcopyq_lane_p8(arg_p8x16, 15, arg_p8x8, 0);
	vcopyq_lane_p8(arg_p8x16, -1, arg_p8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_p8(arg_p8x16, 16, arg_p8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_p8(arg_p8x16, 0, arg_p8x8, 7);
	vcopyq_lane_p8(arg_p8x16, 0, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_p8(arg_p8x16, 0, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_p8(arg_p8x8, 0, arg_p8x16, 0);
	vcopy_laneq_p8(arg_p8x8, 7, arg_p8x16, 0);
	vcopy_laneq_p8(arg_p8x8, -1, arg_p8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_p8(arg_p8x8, 8, arg_p8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_p8(arg_p8x8, 0, arg_p8x16, 15);
	vcopy_laneq_p8(arg_p8x8, 0, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_p8(arg_p8x8, 0, arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_p8(arg_p8x16, 0, arg_p8x16, 0);
	vcopyq_laneq_p8(arg_p8x16, 15, arg_p8x16, 0);
	vcopyq_laneq_p8(arg_p8x16, -1, arg_p8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_p8(arg_p8x16, 16, arg_p8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_p8(arg_p8x16, 0, arg_p8x16, 15);
	vcopyq_laneq_p8(arg_p8x16, 0, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_p8(arg_p8x16, 0, arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_p16(poly16x8_t arg_p16x8, poly16x4_t arg_p16x4) {
	vcopy_lane_p16(arg_p16x4, 0, arg_p16x4, 0);
	vcopy_lane_p16(arg_p16x4, 3, arg_p16x4, 0);
	vcopy_lane_p16(arg_p16x4, -1, arg_p16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_p16(arg_p16x4, 4, arg_p16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_lane_p16(arg_p16x4, 0, arg_p16x4, 3);
	vcopy_lane_p16(arg_p16x4, 0, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_p16(arg_p16x4, 0, arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_p16(arg_p16x8, 0, arg_p16x4, 0);
	vcopyq_lane_p16(arg_p16x8, 7, arg_p16x4, 0);
	vcopyq_lane_p16(arg_p16x8, -1, arg_p16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_p16(arg_p16x8, 8, arg_p16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_p16(arg_p16x8, 0, arg_p16x4, 3);
	vcopyq_lane_p16(arg_p16x8, 0, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_p16(arg_p16x8, 0, arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_p16(arg_p16x4, 0, arg_p16x8, 0);
	vcopy_laneq_p16(arg_p16x4, 3, arg_p16x8, 0);
	vcopy_laneq_p16(arg_p16x4, -1, arg_p16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_p16(arg_p16x4, 4, arg_p16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_p16(arg_p16x4, 0, arg_p16x8, 7);
	vcopy_laneq_p16(arg_p16x4, 0, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_p16(arg_p16x4, 0, arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_p16(arg_p16x8, 0, arg_p16x8, 0);
	vcopyq_laneq_p16(arg_p16x8, 7, arg_p16x8, 0);
	vcopyq_laneq_p16(arg_p16x8, -1, arg_p16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_p16(arg_p16x8, 8, arg_p16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_p16(arg_p16x8, 0, arg_p16x8, 7);
	vcopyq_laneq_p16(arg_p16x8, 0, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_p16(arg_p16x8, 0, arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

