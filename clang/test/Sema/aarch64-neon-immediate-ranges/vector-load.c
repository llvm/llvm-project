// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_vector_load_s8(int8x8x2_t arg_i8x8x2, int8x8x3_t arg_i8x8x3, int8x16x2_t arg_i8x16x2,
						 int8x16x3_t arg_i8x16x3, int8x8_t arg_i8x8, int8x16x4_t arg_i8x16x4,
						 int8x16_t arg_i8x16, int8x8x4_t arg_i8x8x4, int8_t* arg_i8_ptr) {
	vld1_lane_s8(arg_i8_ptr, arg_i8x8, 0);
	vld1_lane_s8(arg_i8_ptr, arg_i8x8, 7);
	vld1_lane_s8(arg_i8_ptr, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_s8(arg_i8_ptr, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_s8(arg_i8_ptr, arg_i8x16, 0);
	vld1q_lane_s8(arg_i8_ptr, arg_i8x16, 15);
	vld1q_lane_s8(arg_i8_ptr, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_s8(arg_i8_ptr, arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_s8(arg_i8_ptr, arg_i8x8x2, 0);
	vld2_lane_s8(arg_i8_ptr, arg_i8x8x2, 7);
	vld2_lane_s8(arg_i8_ptr, arg_i8x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_s8(arg_i8_ptr, arg_i8x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_s8(arg_i8_ptr, arg_i8x16x2, 0);
	vld2q_lane_s8(arg_i8_ptr, arg_i8x16x2, 15);
	vld2q_lane_s8(arg_i8_ptr, arg_i8x16x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_s8(arg_i8_ptr, arg_i8x16x2, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_s8(arg_i8_ptr, arg_i8x8x3, 0);
	vld3_lane_s8(arg_i8_ptr, arg_i8x8x3, 7);
	vld3_lane_s8(arg_i8_ptr, arg_i8x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_s8(arg_i8_ptr, arg_i8x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_s8(arg_i8_ptr, arg_i8x16x3, 0);
	vld3q_lane_s8(arg_i8_ptr, arg_i8x16x3, 15);
	vld3q_lane_s8(arg_i8_ptr, arg_i8x16x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_s8(arg_i8_ptr, arg_i8x16x3, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_s8(arg_i8_ptr, arg_i8x8x4, 0);
	vld4_lane_s8(arg_i8_ptr, arg_i8x8x4, 7);
	vld4_lane_s8(arg_i8_ptr, arg_i8x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_s8(arg_i8_ptr, arg_i8x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_s8(arg_i8_ptr, arg_i8x16x4, 0);
	vld4q_lane_s8(arg_i8_ptr, arg_i8x16x4, 15);
	vld4q_lane_s8(arg_i8_ptr, arg_i8x16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_s8(arg_i8_ptr, arg_i8x16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_s16(int16x8x2_t arg_i16x8x2, int16x8x3_t arg_i16x8x3, int16x8x4_t arg_i16x8x4,
						  int16_t* arg_i16_ptr, int16x4x2_t arg_i16x4x2, int16x4x3_t arg_i16x4x3,
						  int16x8_t arg_i16x8, int16x4x4_t arg_i16x4x4, int16x4_t arg_i16x4) {
	vld1_lane_s16(arg_i16_ptr, arg_i16x4, 0);
	vld1_lane_s16(arg_i16_ptr, arg_i16x4, 3);
	vld1_lane_s16(arg_i16_ptr, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_s16(arg_i16_ptr, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_s16(arg_i16_ptr, arg_i16x8, 0);
	vld1q_lane_s16(arg_i16_ptr, arg_i16x8, 7);
	vld1q_lane_s16(arg_i16_ptr, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_s16(arg_i16_ptr, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_s16(arg_i16_ptr, arg_i16x4x2, 0);
	vld2_lane_s16(arg_i16_ptr, arg_i16x4x2, 3);
	vld2_lane_s16(arg_i16_ptr, arg_i16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_s16(arg_i16_ptr, arg_i16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_s16(arg_i16_ptr, arg_i16x8x2, 0);
	vld2q_lane_s16(arg_i16_ptr, arg_i16x8x2, 7);
	vld2q_lane_s16(arg_i16_ptr, arg_i16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_s16(arg_i16_ptr, arg_i16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_s16(arg_i16_ptr, arg_i16x4x3, 0);
	vld3_lane_s16(arg_i16_ptr, arg_i16x4x3, 3);
	vld3_lane_s16(arg_i16_ptr, arg_i16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_s16(arg_i16_ptr, arg_i16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_s16(arg_i16_ptr, arg_i16x8x3, 0);
	vld3q_lane_s16(arg_i16_ptr, arg_i16x8x3, 7);
	vld3q_lane_s16(arg_i16_ptr, arg_i16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_s16(arg_i16_ptr, arg_i16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_s16(arg_i16_ptr, arg_i16x4x4, 0);
	vld4_lane_s16(arg_i16_ptr, arg_i16x4x4, 3);
	vld4_lane_s16(arg_i16_ptr, arg_i16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_s16(arg_i16_ptr, arg_i16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_s16(arg_i16_ptr, arg_i16x8x4, 0);
	vld4q_lane_s16(arg_i16_ptr, arg_i16x8x4, 7);
	vld4q_lane_s16(arg_i16_ptr, arg_i16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_s16(arg_i16_ptr, arg_i16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_s32(int32x2x4_t arg_i32x2x4, int32x4_t arg_i32x4, int32x2_t arg_i32x2,
						  int32x4x2_t arg_i32x4x2, int32x4x4_t arg_i32x4x4, int32_t* arg_i32_ptr,
						  int32x2x3_t arg_i32x2x3, int32x4x3_t arg_i32x4x3, int32x2x2_t arg_i32x2x2) {
	vld1_lane_s32(arg_i32_ptr, arg_i32x2, 0);
	vld1_lane_s32(arg_i32_ptr, arg_i32x2, 1);
	vld1_lane_s32(arg_i32_ptr, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_s32(arg_i32_ptr, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_s32(arg_i32_ptr, arg_i32x4, 0);
	vld1q_lane_s32(arg_i32_ptr, arg_i32x4, 3);
	vld1q_lane_s32(arg_i32_ptr, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_s32(arg_i32_ptr, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_s32(arg_i32_ptr, arg_i32x2x2, 0);
	vld2_lane_s32(arg_i32_ptr, arg_i32x2x2, 1);
	vld2_lane_s32(arg_i32_ptr, arg_i32x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_s32(arg_i32_ptr, arg_i32x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_s32(arg_i32_ptr, arg_i32x4x2, 0);
	vld2q_lane_s32(arg_i32_ptr, arg_i32x4x2, 3);
	vld2q_lane_s32(arg_i32_ptr, arg_i32x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_s32(arg_i32_ptr, arg_i32x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_s32(arg_i32_ptr, arg_i32x2x3, 0);
	vld3_lane_s32(arg_i32_ptr, arg_i32x2x3, 1);
	vld3_lane_s32(arg_i32_ptr, arg_i32x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_s32(arg_i32_ptr, arg_i32x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_s32(arg_i32_ptr, arg_i32x4x3, 0);
	vld3q_lane_s32(arg_i32_ptr, arg_i32x4x3, 3);
	vld3q_lane_s32(arg_i32_ptr, arg_i32x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_s32(arg_i32_ptr, arg_i32x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_s32(arg_i32_ptr, arg_i32x2x4, 0);
	vld4_lane_s32(arg_i32_ptr, arg_i32x2x4, 1);
	vld4_lane_s32(arg_i32_ptr, arg_i32x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_s32(arg_i32_ptr, arg_i32x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_s32(arg_i32_ptr, arg_i32x4x4, 0);
	vld4q_lane_s32(arg_i32_ptr, arg_i32x4x4, 3);
	vld4q_lane_s32(arg_i32_ptr, arg_i32x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_s32(arg_i32_ptr, arg_i32x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_s64(int64x1x4_t arg_i64x1x4, int64x1_t arg_i64x1, int64x2x2_t arg_i64x2x2,
						  int64x2x4_t arg_i64x2x4, int64x1x3_t arg_i64x1x3, int64x1x2_t arg_i64x1x2,
						  int64x2_t arg_i64x2, int64x2x3_t arg_i64x2x3, int64_t* arg_i64_ptr) {
	vld1_lane_s64(arg_i64_ptr, arg_i64x1, 0);
	vld1_lane_s64(arg_i64_ptr, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_s64(arg_i64_ptr, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_s64(arg_i64_ptr, arg_i64x2, 0);
	vld1q_lane_s64(arg_i64_ptr, arg_i64x2, 1);
	vld1q_lane_s64(arg_i64_ptr, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_s64(arg_i64_ptr, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1_lane_s64(arg_i64_ptr, arg_i64x1, 0);
	vldap1_lane_s64(arg_i64_ptr, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1_lane_s64(arg_i64_ptr, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1q_lane_s64(arg_i64_ptr, arg_i64x2, 0);
	vldap1q_lane_s64(arg_i64_ptr, arg_i64x2, 1);
	vldap1q_lane_s64(arg_i64_ptr, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1q_lane_s64(arg_i64_ptr, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1_lane_s64(arg_i64_ptr, arg_i64x1, 0);
	vstl1_lane_s64(arg_i64_ptr, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1_lane_s64(arg_i64_ptr, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1q_lane_s64(arg_i64_ptr, arg_i64x2, 0);
	vstl1q_lane_s64(arg_i64_ptr, arg_i64x2, 1);
	vstl1q_lane_s64(arg_i64_ptr, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1q_lane_s64(arg_i64_ptr, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_s64(arg_i64_ptr, arg_i64x1x2, 0);
	vld2_lane_s64(arg_i64_ptr, arg_i64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_s64(arg_i64_ptr, arg_i64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_s64(arg_i64_ptr, arg_i64x2x2, 0);
	vld2q_lane_s64(arg_i64_ptr, arg_i64x2x2, 1);
	vld2q_lane_s64(arg_i64_ptr, arg_i64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_s64(arg_i64_ptr, arg_i64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_s64(arg_i64_ptr, arg_i64x1x3, 0);
	vld3_lane_s64(arg_i64_ptr, arg_i64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_s64(arg_i64_ptr, arg_i64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_s64(arg_i64_ptr, arg_i64x2x3, 0);
	vld3q_lane_s64(arg_i64_ptr, arg_i64x2x3, 1);
	vld3q_lane_s64(arg_i64_ptr, arg_i64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_s64(arg_i64_ptr, arg_i64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_s64(arg_i64_ptr, arg_i64x1x4, 0);
	vld4_lane_s64(arg_i64_ptr, arg_i64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_s64(arg_i64_ptr, arg_i64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_s64(arg_i64_ptr, arg_i64x2x4, 0);
	vld4q_lane_s64(arg_i64_ptr, arg_i64x2x4, 1);
	vld4q_lane_s64(arg_i64_ptr, arg_i64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_s64(arg_i64_ptr, arg_i64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_u8(uint8x8x2_t arg_u8x8x2, uint8x16x2_t arg_u8x16x2, uint8x8x4_t arg_u8x8x4,
						uint8x8_t arg_u8x8, uint8x8x3_t arg_u8x8x3, uint8x16_t arg_u8x16,
						uint8x16x4_t arg_u8x16x4, uint8_t *arg_u8_ptr, uint8x16x3_t arg_u8x16x3) {
	vld1_lane_u8(arg_u8_ptr, arg_u8x8, 0);
	vld1_lane_u8(arg_u8_ptr, arg_u8x8, 7);
	vld1_lane_u8(arg_u8_ptr, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_u8(arg_u8_ptr, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_u8(arg_u8_ptr, arg_u8x16, 0);
	vld1q_lane_u8(arg_u8_ptr, arg_u8x16, 15);
	vld1q_lane_u8(arg_u8_ptr, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_u8(arg_u8_ptr, arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_u8(arg_u8_ptr, arg_u8x8x2, 0);
	vld2_lane_u8(arg_u8_ptr, arg_u8x8x2, 7);
	vld2_lane_u8(arg_u8_ptr, arg_u8x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_u8(arg_u8_ptr, arg_u8x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_u8(arg_u8_ptr, arg_u8x16x2, 0);
	vld2q_lane_u8(arg_u8_ptr, arg_u8x16x2, 15);
	vld2q_lane_u8(arg_u8_ptr, arg_u8x16x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_u8(arg_u8_ptr, arg_u8x16x2, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_u8(arg_u8_ptr, arg_u8x8x3, 0);
	vld3_lane_u8(arg_u8_ptr, arg_u8x8x3, 7);
	vld3_lane_u8(arg_u8_ptr, arg_u8x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_u8(arg_u8_ptr, arg_u8x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_u8(arg_u8_ptr, arg_u8x16x3, 0);
	vld3q_lane_u8(arg_u8_ptr, arg_u8x16x3, 15);
	vld3q_lane_u8(arg_u8_ptr, arg_u8x16x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_u8(arg_u8_ptr, arg_u8x16x3, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_u8(arg_u8_ptr, arg_u8x8x4, 0);
	vld4_lane_u8(arg_u8_ptr, arg_u8x8x4, 7);
	vld4_lane_u8(arg_u8_ptr, arg_u8x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_u8(arg_u8_ptr, arg_u8x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_u8(arg_u8_ptr, arg_u8x16x4, 0);
	vld4q_lane_u8(arg_u8_ptr, arg_u8x16x4, 15);
	vld4q_lane_u8(arg_u8_ptr, arg_u8x16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_u8(arg_u8_ptr, arg_u8x16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_u16(uint16x8x2_t arg_u16x8x2, uint16x8x4_t arg_u16x8x4, uint16x4x4_t arg_u16x4x4,
						  uint16x4x2_t arg_u16x4x2, uint16x8_t arg_u16x8, uint16_t *arg_u16_ptr,
						  uint16x8x3_t arg_u16x8x3, uint16x4_t arg_u16x4, uint16x4x3_t arg_u16x4x3) {
	vld1_lane_u16(arg_u16_ptr, arg_u16x4, 0);
	vld1_lane_u16(arg_u16_ptr, arg_u16x4, 3);
	vld1_lane_u16(arg_u16_ptr, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_u16(arg_u16_ptr, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_u16(arg_u16_ptr, arg_u16x8, 0);
	vld1q_lane_u16(arg_u16_ptr, arg_u16x8, 7);
	vld1q_lane_u16(arg_u16_ptr, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_u16(arg_u16_ptr, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_u16(arg_u16_ptr, arg_u16x4x2, 0);
	vld2_lane_u16(arg_u16_ptr, arg_u16x4x2, 3);
	vld2_lane_u16(arg_u16_ptr, arg_u16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_u16(arg_u16_ptr, arg_u16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_u16(arg_u16_ptr, arg_u16x8x2, 0);
	vld2q_lane_u16(arg_u16_ptr, arg_u16x8x2, 7);
	vld2q_lane_u16(arg_u16_ptr, arg_u16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_u16(arg_u16_ptr, arg_u16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_u16(arg_u16_ptr, arg_u16x4x3, 0);
	vld3_lane_u16(arg_u16_ptr, arg_u16x4x3, 3);
	vld3_lane_u16(arg_u16_ptr, arg_u16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_u16(arg_u16_ptr, arg_u16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_u16(arg_u16_ptr, arg_u16x8x3, 0);
	vld3q_lane_u16(arg_u16_ptr, arg_u16x8x3, 7);
	vld3q_lane_u16(arg_u16_ptr, arg_u16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_u16(arg_u16_ptr, arg_u16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_u16(arg_u16_ptr, arg_u16x4x4, 0);
	vld4_lane_u16(arg_u16_ptr, arg_u16x4x4, 3);
	vld4_lane_u16(arg_u16_ptr, arg_u16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_u16(arg_u16_ptr, arg_u16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_u16(arg_u16_ptr, arg_u16x8x4, 0);
	vld4q_lane_u16(arg_u16_ptr, arg_u16x8x4, 7);
	vld4q_lane_u16(arg_u16_ptr, arg_u16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_u16(arg_u16_ptr, arg_u16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_u32(uint32x2x3_t arg_u32x2x3, uint32x2_t arg_u32x2, uint32x2x4_t arg_u32x2x4,
						  uint32x4_t arg_u32x4, uint32x4x2_t arg_u32x4x2, uint32x2x2_t arg_u32x2x2,
						  void *arg_u32_ptr, uint32x4x4_t arg_u32x4x4, uint32x4x3_t arg_u32x4x3) {
	vld1_lane_u32(arg_u32_ptr, arg_u32x2, 0);
	vld1_lane_u32(arg_u32_ptr, arg_u32x2, 1);
	vld1_lane_u32(arg_u32_ptr, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_u32(arg_u32_ptr, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_u32(arg_u32_ptr, arg_u32x4, 0);
	vld1q_lane_u32(arg_u32_ptr, arg_u32x4, 3);
	vld1q_lane_u32(arg_u32_ptr, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_u32(arg_u32_ptr, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_u32(arg_u32_ptr, arg_u32x2x2, 0);
	vld2_lane_u32(arg_u32_ptr, arg_u32x2x2, 1);
	vld2_lane_u32(arg_u32_ptr, arg_u32x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_u32(arg_u32_ptr, arg_u32x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_u32(arg_u32_ptr, arg_u32x4x2, 0);
	vld2q_lane_u32(arg_u32_ptr, arg_u32x4x2, 3);
	vld2q_lane_u32(arg_u32_ptr, arg_u32x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_u32(arg_u32_ptr, arg_u32x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_u32(arg_u32_ptr, arg_u32x2x3, 0);
	vld3_lane_u32(arg_u32_ptr, arg_u32x2x3, 1);
	vld3_lane_u32(arg_u32_ptr, arg_u32x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_u32(arg_u32_ptr, arg_u32x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_u32(arg_u32_ptr, arg_u32x4x3, 0);
	vld3q_lane_u32(arg_u32_ptr, arg_u32x4x3, 3);
	vld3q_lane_u32(arg_u32_ptr, arg_u32x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_u32(arg_u32_ptr, arg_u32x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_u32(arg_u32_ptr, arg_u32x2x4, 0);
	vld4_lane_u32(arg_u32_ptr, arg_u32x2x4, 1);
	vld4_lane_u32(arg_u32_ptr, arg_u32x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_u32(arg_u32_ptr, arg_u32x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_u32(arg_u32_ptr, arg_u32x4x4, 0);
	vld4q_lane_u32(arg_u32_ptr, arg_u32x4x4, 3);
	vld4q_lane_u32(arg_u32_ptr, arg_u32x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_u32(arg_u32_ptr, arg_u32x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_u64(uint64x2x2_t arg_u64x2x2, uint64x1x2_t arg_u64x1x2, uint64x2x3_t arg_u64x2x3,
						  uint64x1_t arg_u64x1, uint64x1x4_t arg_u64x1x4, uint64x1x3_t arg_u64x1x3,
						  uint64_t *arg_u64_ptr, uint64x2_t arg_u64x2, uint64x2x4_t arg_u64x2x4) {
	vld1_lane_u64(arg_u64_ptr, arg_u64x1, 0);
	vld1_lane_u64(arg_u64_ptr, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_u64(arg_u64_ptr, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_u64(arg_u64_ptr, arg_u64x2, 0);
	vld1q_lane_u64(arg_u64_ptr, arg_u64x2, 1);
	vld1q_lane_u64(arg_u64_ptr, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_u64(arg_u64_ptr, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1_lane_u64(arg_u64_ptr, arg_u64x1, 0);
	vldap1_lane_u64(arg_u64_ptr, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1_lane_u64(arg_u64_ptr, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1q_lane_u64(arg_u64_ptr, arg_u64x2, 0);
	vldap1q_lane_u64(arg_u64_ptr, arg_u64x2, 1);
	vldap1q_lane_u64(arg_u64_ptr, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1q_lane_u64(arg_u64_ptr, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1_lane_u64(arg_u64_ptr, arg_u64x1, 0);
	vstl1_lane_u64(arg_u64_ptr, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1_lane_u64(arg_u64_ptr, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1q_lane_u64(arg_u64_ptr, arg_u64x2, 0);
	vstl1q_lane_u64(arg_u64_ptr, arg_u64x2, 1);
	vstl1q_lane_u64(arg_u64_ptr, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1q_lane_u64(arg_u64_ptr, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_u64(arg_u64_ptr, arg_u64x1x2, 0);
	vld2_lane_u64(arg_u64_ptr, arg_u64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_u64(arg_u64_ptr, arg_u64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_u64(arg_u64_ptr, arg_u64x2x2, 0);
	vld2q_lane_u64(arg_u64_ptr, arg_u64x2x2, 1);
	vld2q_lane_u64(arg_u64_ptr, arg_u64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_u64(arg_u64_ptr, arg_u64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_u64(arg_u64_ptr, arg_u64x1x3, 0);
	vld3_lane_u64(arg_u64_ptr, arg_u64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_u64(arg_u64_ptr, arg_u64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_u64(arg_u64_ptr, arg_u64x2x3, 0);
	vld3q_lane_u64(arg_u64_ptr, arg_u64x2x3, 1);
	vld3q_lane_u64(arg_u64_ptr, arg_u64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_u64(arg_u64_ptr, arg_u64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_u64(arg_u64_ptr, arg_u64x1x4, 0);
	vld4_lane_u64(arg_u64_ptr, arg_u64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_u64(arg_u64_ptr, arg_u64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_u64(arg_u64_ptr, arg_u64x2x4, 0);
	vld4q_lane_u64(arg_u64_ptr, arg_u64x2x4, 1);
	vld4q_lane_u64(arg_u64_ptr, arg_u64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_u64(arg_u64_ptr, arg_u64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_p64(poly64_t *arg_p64_ptr, poly64x2x2_t arg_p64x2x2, poly64x1x2_t arg_p64x1x2,
						  poly64x2x4_t arg_p64x2x4, poly64x1x3_t arg_p64x1x3, poly64x2x3_t arg_p64x2x3,
						  poly64x1_t arg_p64x1, poly64x2_t arg_p64x2, poly64x1x4_t arg_p64x1x4) {
	vld1_lane_p64(arg_p64_ptr, arg_p64x1, 0);
	vld1_lane_p64(arg_p64_ptr, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_p64(arg_p64_ptr, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_p64(arg_p64_ptr, arg_p64x2, 0);
	vld1q_lane_p64(arg_p64_ptr, arg_p64x2, 1);
	vld1q_lane_p64(arg_p64_ptr, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_p64(arg_p64_ptr, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1_lane_p64(arg_p64_ptr, arg_p64x1, 0);
	vldap1_lane_p64(arg_p64_ptr, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1_lane_p64(arg_p64_ptr, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1q_lane_p64(arg_p64_ptr, arg_p64x2, 0);
	vldap1q_lane_p64(arg_p64_ptr, arg_p64x2, 1);
	vldap1q_lane_p64(arg_p64_ptr, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1q_lane_p64(arg_p64_ptr, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1_lane_p64(arg_p64_ptr, arg_p64x1, 0);
	vstl1_lane_p64(arg_p64_ptr, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1_lane_p64(arg_p64_ptr, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1q_lane_p64(arg_p64_ptr, arg_p64x2, 0);
	vstl1q_lane_p64(arg_p64_ptr, arg_p64x2, 1);
	vstl1q_lane_p64(arg_p64_ptr, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1q_lane_p64(arg_p64_ptr, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_p64(arg_p64_ptr, arg_p64x1x2, 0);
	vld2_lane_p64(arg_p64_ptr, arg_p64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_p64(arg_p64_ptr, arg_p64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_p64(arg_p64_ptr, arg_p64x2x2, 0);
	vld2q_lane_p64(arg_p64_ptr, arg_p64x2x2, 1);
	vld2q_lane_p64(arg_p64_ptr, arg_p64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_p64(arg_p64_ptr, arg_p64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_p64(arg_p64_ptr, arg_p64x1x3, 0);
	vld3_lane_p64(arg_p64_ptr, arg_p64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_p64(arg_p64_ptr, arg_p64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_p64(arg_p64_ptr, arg_p64x2x3, 0);
	vld3q_lane_p64(arg_p64_ptr, arg_p64x2x3, 1);
	vld3q_lane_p64(arg_p64_ptr, arg_p64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_p64(arg_p64_ptr, arg_p64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_p64(arg_p64_ptr, arg_p64x1x4, 0);
	vld4_lane_p64(arg_p64_ptr, arg_p64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_p64(arg_p64_ptr, arg_p64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_p64(arg_p64_ptr, arg_p64x2x4, 0);
	vld4q_lane_p64(arg_p64_ptr, arg_p64x2x4, 1);
	vld4q_lane_p64(arg_p64_ptr, arg_p64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_p64(arg_p64_ptr, arg_p64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_f16(float16_t *arg_f16_ptr, float16x8_t arg_f16x8, float16x8x2_t arg_f16x8x2,
						  float16x8x3_t arg_f16x8x3, float16x4x4_t arg_f16x4x4, float16x8x4_t arg_f16x8x4,
						  float16x4x2_t arg_f16x4x2, float16x4_t arg_f16x4, float16x4x3_t arg_f16x4x3) {
	vld1_lane_f16(arg_f16_ptr, arg_f16x4, 0);
	vld1_lane_f16(arg_f16_ptr, arg_f16x4, 3);
	vld1_lane_f16(arg_f16_ptr, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_f16(arg_f16_ptr, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_f16(arg_f16_ptr, arg_f16x8, 0);
	vld1q_lane_f16(arg_f16_ptr, arg_f16x8, 7);
	vld1q_lane_f16(arg_f16_ptr, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_f16(arg_f16_ptr, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_f16(arg_f16_ptr, arg_f16x4x2, 0);
	vld2_lane_f16(arg_f16_ptr, arg_f16x4x2, 3);
	vld2_lane_f16(arg_f16_ptr, arg_f16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_f16(arg_f16_ptr, arg_f16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_f16(arg_f16_ptr, arg_f16x8x2, 0);
	vld2q_lane_f16(arg_f16_ptr, arg_f16x8x2, 7);
	vld2q_lane_f16(arg_f16_ptr, arg_f16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_f16(arg_f16_ptr, arg_f16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_f16(arg_f16_ptr, arg_f16x4x3, 0);
	vld3_lane_f16(arg_f16_ptr, arg_f16x4x3, 3);
	vld3_lane_f16(arg_f16_ptr, arg_f16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_f16(arg_f16_ptr, arg_f16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_f16(arg_f16_ptr, arg_f16x8x3, 0);
	vld3q_lane_f16(arg_f16_ptr, arg_f16x8x3, 7);
	vld3q_lane_f16(arg_f16_ptr, arg_f16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_f16(arg_f16_ptr, arg_f16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_f16(arg_f16_ptr, arg_f16x4x4, 0);
	vld4_lane_f16(arg_f16_ptr, arg_f16x4x4, 3);
	vld4_lane_f16(arg_f16_ptr, arg_f16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_f16(arg_f16_ptr, arg_f16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_f16(arg_f16_ptr, arg_f16x8x4, 0);
	vld4q_lane_f16(arg_f16_ptr, arg_f16x8x4, 7);
	vld4q_lane_f16(arg_f16_ptr, arg_f16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_f16(arg_f16_ptr, arg_f16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_f32(float32_t *arg_f32_ptr, float32x4x3_t arg_f32x4x3, float32x2x4_t arg_f32x2x4,
						  float32x4x4_t arg_f32x4x4, float32x2x3_t arg_f32x2x3, float32x2x2_t arg_f32x2x2,
						  float32x4x2_t arg_f32x4x2, float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vld1_lane_f32(arg_f32_ptr, arg_f32x2, 0);
	vld1_lane_f32(arg_f32_ptr, arg_f32x2, 1);
	vld1_lane_f32(arg_f32_ptr, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_f32(arg_f32_ptr, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_f32(arg_f32_ptr, arg_f32x4, 0);
	vld1q_lane_f32(arg_f32_ptr, arg_f32x4, 3);
	vld1q_lane_f32(arg_f32_ptr, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_f32(arg_f32_ptr, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_f32(arg_f32_ptr, arg_f32x2x2, 0);
	vld2_lane_f32(arg_f32_ptr, arg_f32x2x2, 1);
	vld2_lane_f32(arg_f32_ptr, arg_f32x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_f32(arg_f32_ptr, arg_f32x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_f32(arg_f32_ptr, arg_f32x4x2, 0);
	vld2q_lane_f32(arg_f32_ptr, arg_f32x4x2, 3);
	vld2q_lane_f32(arg_f32_ptr, arg_f32x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_f32(arg_f32_ptr, arg_f32x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_f32(arg_f32_ptr, arg_f32x2x3, 0);
	vld3_lane_f32(arg_f32_ptr, arg_f32x2x3, 1);
	vld3_lane_f32(arg_f32_ptr, arg_f32x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_f32(arg_f32_ptr, arg_f32x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_f32(arg_f32_ptr, arg_f32x4x3, 0);
	vld3q_lane_f32(arg_f32_ptr, arg_f32x4x3, 3);
	vld3q_lane_f32(arg_f32_ptr, arg_f32x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_f32(arg_f32_ptr, arg_f32x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_f32(arg_f32_ptr, arg_f32x2x4, 0);
	vld4_lane_f32(arg_f32_ptr, arg_f32x2x4, 1);
	vld4_lane_f32(arg_f32_ptr, arg_f32x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_f32(arg_f32_ptr, arg_f32x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_f32(arg_f32_ptr, arg_f32x4x4, 0);
	vld4q_lane_f32(arg_f32_ptr, arg_f32x4x4, 3);
	vld4q_lane_f32(arg_f32_ptr, arg_f32x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_f32(arg_f32_ptr, arg_f32x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_p8(poly8x16_t arg_p8x16, poly8x8x2_t arg_p8x8x2, poly8x16x4_t arg_p8x16x4,
						 poly8_t *arg_p8_ptr, poly8x8_t arg_p8x8, poly8x8x4_t arg_p8x8x4,
						 poly8x16x2_t arg_p8x16x2, poly8x8x3_t arg_p8x8x3, poly8x16x3_t arg_p8x16x3) {
	vld1_lane_p8(arg_p8_ptr, arg_p8x8, 0);
	vld1_lane_p8(arg_p8_ptr, arg_p8x8, 7);
	vld1_lane_p8(arg_p8_ptr, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_p8(arg_p8_ptr, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_p8(arg_p8_ptr, arg_p8x16, 0);
	vld1q_lane_p8(arg_p8_ptr, arg_p8x16, 15);
	vld1q_lane_p8(arg_p8_ptr, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_p8(arg_p8_ptr, arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_p8(arg_p8_ptr, arg_p8x8x2, 0);
	vld2_lane_p8(arg_p8_ptr, arg_p8x8x2, 7);
	vld2_lane_p8(arg_p8_ptr, arg_p8x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_p8(arg_p8_ptr, arg_p8x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_p8(arg_p8_ptr, arg_p8x16x2, 0);
	vld2q_lane_p8(arg_p8_ptr, arg_p8x16x2, 15);
	vld2q_lane_p8(arg_p8_ptr, arg_p8x16x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_p8(arg_p8_ptr, arg_p8x16x2, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_p8(arg_p8_ptr, arg_p8x8x3, 0);
	vld3_lane_p8(arg_p8_ptr, arg_p8x8x3, 7);
	vld3_lane_p8(arg_p8_ptr, arg_p8x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_p8(arg_p8_ptr, arg_p8x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_p8(arg_p8_ptr, arg_p8x16x3, 0);
	vld3q_lane_p8(arg_p8_ptr, arg_p8x16x3, 15);
	vld3q_lane_p8(arg_p8_ptr, arg_p8x16x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_p8(arg_p8_ptr, arg_p8x16x3, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_p8(arg_p8_ptr, arg_p8x8x4, 0);
	vld4_lane_p8(arg_p8_ptr, arg_p8x8x4, 7);
	vld4_lane_p8(arg_p8_ptr, arg_p8x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_p8(arg_p8_ptr, arg_p8x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_p8(arg_p8_ptr, arg_p8x16x4, 0);
	vld4q_lane_p8(arg_p8_ptr, arg_p8x16x4, 15);
	vld4q_lane_p8(arg_p8_ptr, arg_p8x16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_p8(arg_p8_ptr, arg_p8x16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_p16(poly16x8x4_t arg_p16x8x4, poly16x8_t arg_p16x8, poly16x4x4_t arg_p16x4x4,
						  poly16x8x3_t arg_p16x8x3, poly16_t *arg_p16_ptr, poly16x4_t arg_p16x4,
						  poly16x8x2_t arg_p16x8x2, poly16x4x2_t arg_p16x4x2, poly16x4x3_t arg_p16x4x3) {
	vld1_lane_p16(arg_p16_ptr, arg_p16x4, 0);
	vld1_lane_p16(arg_p16_ptr, arg_p16x4, 3);
	vld1_lane_p16(arg_p16_ptr, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_p16(arg_p16_ptr, arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_p16(arg_p16_ptr, arg_p16x8, 0);
	vld1q_lane_p16(arg_p16_ptr, arg_p16x8, 7);
	vld1q_lane_p16(arg_p16_ptr, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_p16(arg_p16_ptr, arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_p16(arg_p16_ptr, arg_p16x4x2, 0);
	vld2_lane_p16(arg_p16_ptr, arg_p16x4x2, 3);
	vld2_lane_p16(arg_p16_ptr, arg_p16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_p16(arg_p16_ptr, arg_p16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_p16(arg_p16_ptr, arg_p16x8x2, 0);
	vld2q_lane_p16(arg_p16_ptr, arg_p16x8x2, 7);
	vld2q_lane_p16(arg_p16_ptr, arg_p16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_p16(arg_p16_ptr, arg_p16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_p16(arg_p16_ptr, arg_p16x4x3, 0);
	vld3_lane_p16(arg_p16_ptr, arg_p16x4x3, 3);
	vld3_lane_p16(arg_p16_ptr, arg_p16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_p16(arg_p16_ptr, arg_p16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_p16(arg_p16_ptr, arg_p16x8x3, 0);
	vld3q_lane_p16(arg_p16_ptr, arg_p16x8x3, 7);
	vld3q_lane_p16(arg_p16_ptr, arg_p16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_p16(arg_p16_ptr, arg_p16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_p16(arg_p16_ptr, arg_p16x4x4, 0);
	vld4_lane_p16(arg_p16_ptr, arg_p16x4x4, 3);
	vld4_lane_p16(arg_p16_ptr, arg_p16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_p16(arg_p16_ptr, arg_p16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_p16(arg_p16_ptr, arg_p16x8x4, 0);
	vld4q_lane_p16(arg_p16_ptr, arg_p16x8x4, 7);
	vld4q_lane_p16(arg_p16_ptr, arg_p16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_p16(arg_p16_ptr, arg_p16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_load_f64(float64x1_t arg_f64x1, float64x1x2_t arg_f64x1x2, float64_t* arg_f64_ptr,
						  float64x2x3_t arg_f64x2x3, float64x2x4_t arg_f64x2x4, float64x2x2_t arg_f64x2x2,
						  float64x2_t arg_f64x2, float64x1x3_t arg_f64x1x3, float64x1x4_t arg_f64x1x4) {
	vld1_lane_f64(arg_f64_ptr, arg_f64x1, 0);
	vld1_lane_f64(arg_f64_ptr, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_f64(arg_f64_ptr, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_f64(arg_f64_ptr, arg_f64x2, 0);
	vld1q_lane_f64(arg_f64_ptr, arg_f64x2, 1);
	vld1q_lane_f64(arg_f64_ptr, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_f64(arg_f64_ptr, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1_lane_f64(arg_f64_ptr, arg_f64x1, 0);
	vldap1_lane_f64(arg_f64_ptr, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1_lane_f64(arg_f64_ptr, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vldap1q_lane_f64(arg_f64_ptr, arg_f64x2, 0);
	vldap1q_lane_f64(arg_f64_ptr, arg_f64x2, 1);
	vldap1q_lane_f64(arg_f64_ptr, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vldap1q_lane_f64(arg_f64_ptr, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1_lane_f64(arg_f64_ptr, arg_f64x1, 0);
	vstl1_lane_f64(arg_f64_ptr, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1_lane_f64(arg_f64_ptr, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vstl1q_lane_f64(arg_f64_ptr, arg_f64x2, 0);
	vstl1q_lane_f64(arg_f64_ptr, arg_f64x2, 1);
	vstl1q_lane_f64(arg_f64_ptr, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vstl1q_lane_f64(arg_f64_ptr, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_f64(arg_f64_ptr, arg_f64x1x2, 0);
	vld2_lane_f64(arg_f64_ptr, arg_f64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_f64(arg_f64_ptr, arg_f64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_f64(arg_f64_ptr, arg_f64x2x2, 0);
	vld2q_lane_f64(arg_f64_ptr, arg_f64x2x2, 1);
	vld2q_lane_f64(arg_f64_ptr, arg_f64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_f64(arg_f64_ptr, arg_f64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_f64(arg_f64_ptr, arg_f64x1x3, 0);
	vld3_lane_f64(arg_f64_ptr, arg_f64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_f64(arg_f64_ptr, arg_f64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_f64(arg_f64_ptr, arg_f64x2x3, 0);
	vld3q_lane_f64(arg_f64_ptr, arg_f64x2x3, 1);
	vld3q_lane_f64(arg_f64_ptr, arg_f64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_f64(arg_f64_ptr, arg_f64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_f64(arg_f64_ptr, arg_f64x1x4, 0);
	vld4_lane_f64(arg_f64_ptr, arg_f64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_f64(arg_f64_ptr, arg_f64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_f64(arg_f64_ptr, arg_f64x2x4, 0);
	vld4q_lane_f64(arg_f64_ptr, arg_f64x2x4, 1);
	vld4q_lane_f64(arg_f64_ptr, arg_f64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_f64(arg_f64_ptr, arg_f64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

