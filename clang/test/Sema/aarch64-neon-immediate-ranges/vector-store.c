// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_vector_store_s8(int8x8_t arg_i8x8, int8x8x3_t arg_i8x8x3, int8_t* arg_i8_ptr,
						  int8x16x3_t arg_i8x16x3,  int8x8x4_t arg_i8x8x4, int8x16x4_t arg_i8x16x4,
						  int8x8x2_t arg_i8x8x2, int8x16_t arg_i8x16, int8x16x2_t arg_i8x16x2) {
	vst1_lane_s8(arg_i8_ptr, arg_i8x8, 0);
	vst1_lane_s8(arg_i8_ptr, arg_i8x8, 7);
	vst1_lane_s8(arg_i8_ptr, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_s8(arg_i8_ptr, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_s8(arg_i8_ptr, arg_i8x16, 0);
	vst1q_lane_s8(arg_i8_ptr, arg_i8x16, 15);
	vst1q_lane_s8(arg_i8_ptr, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_s8(arg_i8_ptr, arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_s8(arg_i8_ptr, arg_i8x8x2, 0);
	vst2_lane_s8(arg_i8_ptr, arg_i8x8x2, 7);
	vst2_lane_s8(arg_i8_ptr, arg_i8x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_s8(arg_i8_ptr, arg_i8x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_s8(arg_i8_ptr, arg_i8x8x3, 0);
	vst3_lane_s8(arg_i8_ptr, arg_i8x8x3, 7);
	vst3_lane_s8(arg_i8_ptr, arg_i8x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_s8(arg_i8_ptr, arg_i8x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_s8(arg_i8_ptr, arg_i8x8x4, 0);
	vst4_lane_s8(arg_i8_ptr, arg_i8x8x4, 7);
	vst4_lane_s8(arg_i8_ptr, arg_i8x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_s8(arg_i8_ptr, arg_i8x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_s8(arg_i8_ptr, arg_i8x16x2, 0);
	vst2q_lane_s8(arg_i8_ptr, arg_i8x16x2, 15);
	vst2q_lane_s8(arg_i8_ptr, arg_i8x16x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_s8(arg_i8_ptr, arg_i8x16x2, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_s8(arg_i8_ptr, arg_i8x16x3, 0);
	vst3q_lane_s8(arg_i8_ptr, arg_i8x16x3, 15);
	vst3q_lane_s8(arg_i8_ptr, arg_i8x16x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_s8(arg_i8_ptr, arg_i8x16x3, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_s8(arg_i8_ptr, arg_i8x16x4, 0);
	vst4q_lane_s8(arg_i8_ptr, arg_i8x16x4, 15);
	vst4q_lane_s8(arg_i8_ptr, arg_i8x16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_s8(arg_i8_ptr, arg_i8x16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_s16(int16x8x3_t arg_i16x8x3, int16x4_t arg_i16x4, int16x4x3_t arg_i16x4x3,
						   int16x8_t arg_i16x8, int16_t* arg_i16_ptr, int16x8x2_t arg_i16x8x2,
						   int16x8x4_t arg_i16x8x4, int16x4x4_t arg_i16x4x4, int16x4x2_t arg_i16x4x2) {
	vst1_lane_s16(arg_i16_ptr, arg_i16x4, 0);
	vst1_lane_s16(arg_i16_ptr, arg_i16x4, 3);
	vst1_lane_s16(arg_i16_ptr, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_s16(arg_i16_ptr, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_s16(arg_i16_ptr, arg_i16x8, 0);
	vst1q_lane_s16(arg_i16_ptr, arg_i16x8, 7);
	vst1q_lane_s16(arg_i16_ptr, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_s16(arg_i16_ptr, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_s16(arg_i16_ptr, arg_i16x4x2, 0);
	vst2_lane_s16(arg_i16_ptr, arg_i16x4x2, 3);
	vst2_lane_s16(arg_i16_ptr, arg_i16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_s16(arg_i16_ptr, arg_i16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_s16(arg_i16_ptr, arg_i16x8x2, 0);
	vst2q_lane_s16(arg_i16_ptr, arg_i16x8x2, 7);
	vst2q_lane_s16(arg_i16_ptr, arg_i16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_s16(arg_i16_ptr, arg_i16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_s16(arg_i16_ptr, arg_i16x4x3, 0);
	vst3_lane_s16(arg_i16_ptr, arg_i16x4x3, 3);
	vst3_lane_s16(arg_i16_ptr, arg_i16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_s16(arg_i16_ptr, arg_i16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_s16(arg_i16_ptr, arg_i16x8x3, 0);
	vst3q_lane_s16(arg_i16_ptr, arg_i16x8x3, 7);
	vst3q_lane_s16(arg_i16_ptr, arg_i16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_s16(arg_i16_ptr, arg_i16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_s16(arg_i16_ptr, arg_i16x4x4, 0);
	vst4_lane_s16(arg_i16_ptr, arg_i16x4x4, 3);
	vst4_lane_s16(arg_i16_ptr, arg_i16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_s16(arg_i16_ptr, arg_i16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_s16(arg_i16_ptr, arg_i16x8x4, 0);
	vst4q_lane_s16(arg_i16_ptr, arg_i16x8x4, 7);
	vst4q_lane_s16(arg_i16_ptr, arg_i16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_s16(arg_i16_ptr, arg_i16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_s32(int32x4x3_t arg_i32x4x3, int32x4_t arg_i32x4, int32x2x2_t arg_i32x2x2,
						   int32x2x3_t arg_i32x2x3, int32x4x4_t arg_i32x4x4, int32x4x2_t arg_i32x4x2,
						   int32x2_t arg_i32x2, int32x2x4_t arg_i32x2x4, int32_t* arg_i32_ptr) {
	vst1_lane_s32(arg_i32_ptr, arg_i32x2, 0);
	vst1_lane_s32(arg_i32_ptr, arg_i32x2, 1);
	vst1_lane_s32(arg_i32_ptr, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_s32(arg_i32_ptr, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_s32(arg_i32_ptr, arg_i32x4, 0);
	vst1q_lane_s32(arg_i32_ptr, arg_i32x4, 3);
	vst1q_lane_s32(arg_i32_ptr, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_s32(arg_i32_ptr, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_s32(arg_i32_ptr, arg_i32x2x2, 0);
	vst2_lane_s32(arg_i32_ptr, arg_i32x2x2, 1);
	vst2_lane_s32(arg_i32_ptr, arg_i32x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_s32(arg_i32_ptr, arg_i32x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_s32(arg_i32_ptr, arg_i32x4x2, 0);
	vst2q_lane_s32(arg_i32_ptr, arg_i32x4x2, 3);
	vst2q_lane_s32(arg_i32_ptr, arg_i32x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_s32(arg_i32_ptr, arg_i32x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_s32(arg_i32_ptr, arg_i32x2x3, 0);
	vst3_lane_s32(arg_i32_ptr, arg_i32x2x3, 1);
	vst3_lane_s32(arg_i32_ptr, arg_i32x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_s32(arg_i32_ptr, arg_i32x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_s32(arg_i32_ptr, arg_i32x4x3, 0);
	vst3q_lane_s32(arg_i32_ptr, arg_i32x4x3, 3);
	vst3q_lane_s32(arg_i32_ptr, arg_i32x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_s32(arg_i32_ptr, arg_i32x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_s32(arg_i32_ptr, arg_i32x2x4, 0);
	vst4_lane_s32(arg_i32_ptr, arg_i32x2x4, 1);
	vst4_lane_s32(arg_i32_ptr, arg_i32x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_s32(arg_i32_ptr, arg_i32x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_s32(arg_i32_ptr, arg_i32x4x4, 0);
	vst4q_lane_s32(arg_i32_ptr, arg_i32x4x4, 3);
	vst4q_lane_s32(arg_i32_ptr, arg_i32x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_s32(arg_i32_ptr, arg_i32x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_s64(int64x2x2_t arg_i64x2x2, int64_t* arg_i64_ptr, int64x1_t arg_i64x1,
						   int64x2x4_t arg_i64x2x4, int64x1x4_t arg_i64x1x4, int64x1x2_t arg_i64x1x2,
						   int64x1x3_t arg_i64x1x3, int64x2x3_t arg_i64x2x3, int64x2_t arg_i64x2) {
	vst1_lane_s64(arg_i64_ptr, arg_i64x1, 0);
	vst1_lane_s64(arg_i64_ptr, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_s64(arg_i64_ptr, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_s64(arg_i64_ptr, arg_i64x2, 0);
	vst1q_lane_s64(arg_i64_ptr, arg_i64x2, 1);
	vst1q_lane_s64(arg_i64_ptr, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_s64(arg_i64_ptr, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_s64(arg_i64_ptr, arg_i64x1x2, 0);
	vst2_lane_s64(arg_i64_ptr, arg_i64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_s64(arg_i64_ptr, arg_i64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_s64(arg_i64_ptr, arg_i64x2x2, 0);
	vst2q_lane_s64(arg_i64_ptr, arg_i64x2x2, 1);
	vst2q_lane_s64(arg_i64_ptr, arg_i64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_s64(arg_i64_ptr, arg_i64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_s64(arg_i64_ptr, arg_i64x1x3, 0);
	vst3_lane_s64(arg_i64_ptr, arg_i64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_s64(arg_i64_ptr, arg_i64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_s64(arg_i64_ptr, arg_i64x2x3, 0);
	vst3q_lane_s64(arg_i64_ptr, arg_i64x2x3, 1);
	vst3q_lane_s64(arg_i64_ptr, arg_i64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_s64(arg_i64_ptr, arg_i64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_s64(arg_i64_ptr, arg_i64x1x4, 0);
	vst4_lane_s64(arg_i64_ptr, arg_i64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_s64(arg_i64_ptr, arg_i64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_s64(arg_i64_ptr, arg_i64x2x4, 0);
	vst4q_lane_s64(arg_i64_ptr, arg_i64x2x4, 1);
	vst4q_lane_s64(arg_i64_ptr, arg_i64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_s64(arg_i64_ptr, arg_i64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_u8(uint8x16_t arg_u8x16, uint8x16x3_t arg_u8x16x3, uint8x8x4_t arg_u8x8x4,
						  uint8x16x2_t arg_u8x16x2, uint8x8_t arg_u8x8, uint8x8x3_t arg_u8x8x3,
						  uint8x16x4_t arg_u8x16x4, uint8_t* arg_u8_ptr, uint8x8x2_t arg_u8x8x2) {
	vst1_lane_u8(arg_u8_ptr, arg_u8x8, 0);
	vst1_lane_u8(arg_u8_ptr, arg_u8x8, 7);
	vst1_lane_u8(arg_u8_ptr, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_u8(arg_u8_ptr, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_u8(arg_u8_ptr, arg_u8x16, 0);
	vst1q_lane_u8(arg_u8_ptr, arg_u8x16, 15);
	vst1q_lane_u8(arg_u8_ptr, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_u8(arg_u8_ptr, arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_u8(arg_u8_ptr, arg_u8x8x2, 0);
	vst2_lane_u8(arg_u8_ptr, arg_u8x8x2, 7);
	vst2_lane_u8(arg_u8_ptr, arg_u8x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_u8(arg_u8_ptr, arg_u8x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_u8(arg_u8_ptr, arg_u8x8x3, 0);
	vst3_lane_u8(arg_u8_ptr, arg_u8x8x3, 7);
	vst3_lane_u8(arg_u8_ptr, arg_u8x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_u8(arg_u8_ptr, arg_u8x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_u8(arg_u8_ptr, arg_u8x8x4, 0);
	vst4_lane_u8(arg_u8_ptr, arg_u8x8x4, 7);
	vst4_lane_u8(arg_u8_ptr, arg_u8x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_u8(arg_u8_ptr, arg_u8x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_u8(arg_u8_ptr, arg_u8x16x2, 0);
	vst2q_lane_u8(arg_u8_ptr, arg_u8x16x2, 15);
	vst2q_lane_u8(arg_u8_ptr, arg_u8x16x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_u8(arg_u8_ptr, arg_u8x16x2, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_u8(arg_u8_ptr, arg_u8x16x3, 0);
	vst3q_lane_u8(arg_u8_ptr, arg_u8x16x3, 15);
	vst3q_lane_u8(arg_u8_ptr, arg_u8x16x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_u8(arg_u8_ptr, arg_u8x16x3, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_u8(arg_u8_ptr, arg_u8x16x4, 0);
	vst4q_lane_u8(arg_u8_ptr, arg_u8x16x4, 15);
	vst4q_lane_u8(arg_u8_ptr, arg_u8x16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_u8(arg_u8_ptr, arg_u8x16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_u16(uint16x8x3_t arg_u16x8x3, uint16x4x4_t arg_u16x4x4, uint16_t* arg_u16_ptr,
						   uint16x4_t arg_u16x4, uint16x4x2_t arg_u16x4x2, uint16x4x3_t arg_u16x4x3,
						   uint16x8_t arg_u16x8, uint16x8x2_t arg_u16x8x2, uint16x8x4_t arg_u16x8x4) {
	vst1_lane_u16(arg_u16_ptr, arg_u16x4, 0);
	vst1_lane_u16(arg_u16_ptr, arg_u16x4, 3);
	vst1_lane_u16(arg_u16_ptr, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_u16(arg_u16_ptr, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_u16(arg_u16_ptr, arg_u16x8, 0);
	vst1q_lane_u16(arg_u16_ptr, arg_u16x8, 7);
	vst1q_lane_u16(arg_u16_ptr, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_u16(arg_u16_ptr, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_u16(arg_u16_ptr, arg_u16x4x2, 0);
	vst2_lane_u16(arg_u16_ptr, arg_u16x4x2, 3);
	vst2_lane_u16(arg_u16_ptr, arg_u16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_u16(arg_u16_ptr, arg_u16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_u16(arg_u16_ptr, arg_u16x8x2, 0);
	vst2q_lane_u16(arg_u16_ptr, arg_u16x8x2, 7);
	vst2q_lane_u16(arg_u16_ptr, arg_u16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_u16(arg_u16_ptr, arg_u16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_u16(arg_u16_ptr, arg_u16x4x3, 0);
	vst3_lane_u16(arg_u16_ptr, arg_u16x4x3, 3);
	vst3_lane_u16(arg_u16_ptr, arg_u16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_u16(arg_u16_ptr, arg_u16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_u16(arg_u16_ptr, arg_u16x8x3, 0);
	vst3q_lane_u16(arg_u16_ptr, arg_u16x8x3, 7);
	vst3q_lane_u16(arg_u16_ptr, arg_u16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_u16(arg_u16_ptr, arg_u16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_u16(arg_u16_ptr, arg_u16x4x4, 0);
	vst4_lane_u16(arg_u16_ptr, arg_u16x4x4, 3);
	vst4_lane_u16(arg_u16_ptr, arg_u16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_u16(arg_u16_ptr, arg_u16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_u16(arg_u16_ptr, arg_u16x8x4, 0);
	vst4q_lane_u16(arg_u16_ptr, arg_u16x8x4, 7);
	vst4q_lane_u16(arg_u16_ptr, arg_u16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_u16(arg_u16_ptr, arg_u16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_u32(uint32x4x3_t arg_u32x4x3, uint32x2_t arg_u32x2, uint32x2x3_t arg_u32x2x3,
						   uint32x4x4_t arg_u32x4x4, uint32x4_t arg_u32x4, uint32x2x2_t arg_u32x2x2,
						   uint32_t* arg_u32_ptr, uint32x2x4_t arg_u32x2x4, uint32x4x2_t arg_u32x4x2) {
	vst1_lane_u32(arg_u32_ptr, arg_u32x2, 0);
	vst1_lane_u32(arg_u32_ptr, arg_u32x2, 1);
	vst1_lane_u32(arg_u32_ptr, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_u32(arg_u32_ptr, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_u32(arg_u32_ptr, arg_u32x4, 0);
	vst1q_lane_u32(arg_u32_ptr, arg_u32x4, 3);
	vst1q_lane_u32(arg_u32_ptr, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_u32(arg_u32_ptr, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_u32(arg_u32_ptr, arg_u32x2x2, 0);
	vst2_lane_u32(arg_u32_ptr, arg_u32x2x2, 1);
	vst2_lane_u32(arg_u32_ptr, arg_u32x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_u32(arg_u32_ptr, arg_u32x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_u32(arg_u32_ptr, arg_u32x4x2, 0);
	vst2q_lane_u32(arg_u32_ptr, arg_u32x4x2, 3);
	vst2q_lane_u32(arg_u32_ptr, arg_u32x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_u32(arg_u32_ptr, arg_u32x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_u32(arg_u32_ptr, arg_u32x2x3, 0);
	vst3_lane_u32(arg_u32_ptr, arg_u32x2x3, 1);
	vst3_lane_u32(arg_u32_ptr, arg_u32x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_u32(arg_u32_ptr, arg_u32x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_u32(arg_u32_ptr, arg_u32x4x3, 0);
	vst3q_lane_u32(arg_u32_ptr, arg_u32x4x3, 3);
	vst3q_lane_u32(arg_u32_ptr, arg_u32x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_u32(arg_u32_ptr, arg_u32x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_u32(arg_u32_ptr, arg_u32x2x4, 0);
	vst4_lane_u32(arg_u32_ptr, arg_u32x2x4, 1);
	vst4_lane_u32(arg_u32_ptr, arg_u32x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_u32(arg_u32_ptr, arg_u32x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_u32(arg_u32_ptr, arg_u32x4x4, 0);
	vst4q_lane_u32(arg_u32_ptr, arg_u32x4x4, 3);
	vst4q_lane_u32(arg_u32_ptr, arg_u32x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_u32(arg_u32_ptr, arg_u32x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_u64(uint64x2x3_t arg_u64x2x3, uint64x1_t arg_u64x1, uint64x2_t arg_u64x2,
						   uint64x1x2_t arg_u64x1x2, uint64x2x2_t arg_u64x2x2, uint64x1x3_t arg_u64x1x3,
						   uint64_t* arg_u64_ptr, uint64x2x4_t arg_u64x2x4, uint64x1x4_t arg_u64x1x4) {
	vst1_lane_u64(arg_u64_ptr, arg_u64x1, 0);
	vst1_lane_u64(arg_u64_ptr, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_u64(arg_u64_ptr, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_u64(arg_u64_ptr, arg_u64x2, 0);
	vst1q_lane_u64(arg_u64_ptr, arg_u64x2, 1);
	vst1q_lane_u64(arg_u64_ptr, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_u64(arg_u64_ptr, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_u64(arg_u64_ptr, arg_u64x1x2, 0);
	vst2_lane_u64(arg_u64_ptr, arg_u64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_u64(arg_u64_ptr, arg_u64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_u64(arg_u64_ptr, arg_u64x2x2, 0);
	vst2q_lane_u64(arg_u64_ptr, arg_u64x2x2, 1);
	vst2q_lane_u64(arg_u64_ptr, arg_u64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_u64(arg_u64_ptr, arg_u64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_u64(arg_u64_ptr, arg_u64x1x3, 0);
	vst3_lane_u64(arg_u64_ptr, arg_u64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_u64(arg_u64_ptr, arg_u64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_u64(arg_u64_ptr, arg_u64x2x3, 0);
	vst3q_lane_u64(arg_u64_ptr, arg_u64x2x3, 1);
	vst3q_lane_u64(arg_u64_ptr, arg_u64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_u64(arg_u64_ptr, arg_u64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_u64(arg_u64_ptr, arg_u64x1x4, 0);
	vst4_lane_u64(arg_u64_ptr, arg_u64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_u64(arg_u64_ptr, arg_u64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_u64(arg_u64_ptr, arg_u64x2x4, 0);
	vst4q_lane_u64(arg_u64_ptr, arg_u64x2x4, 1);
	vst4q_lane_u64(arg_u64_ptr, arg_u64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_u64(arg_u64_ptr, arg_u64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_p64(poly64x2x4_t arg_p64x2x4, poly64x1x3_t arg_p64x1x3, poly64x1_t arg_p64x1,
						   poly64x2x2_t arg_p64x2x2, poly64x1x4_t arg_p64x1x4, poly64_t* arg_p64_ptr,
						   poly64x1x2_t arg_p64x1x2, poly64x2_t arg_p64x2, poly64x2x3_t arg_p64x2x3) {
	vst1_lane_p64(arg_p64_ptr, arg_p64x1, 0);
	vst1_lane_p64(arg_p64_ptr, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_p64(arg_p64_ptr, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_p64(arg_p64_ptr, arg_p64x2, 0);
	vst1q_lane_p64(arg_p64_ptr, arg_p64x2, 1);
	vst1q_lane_p64(arg_p64_ptr, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_p64(arg_p64_ptr, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_p64(arg_p64_ptr, arg_p64x1x2, 0);
	vst2_lane_p64(arg_p64_ptr, arg_p64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_p64(arg_p64_ptr, arg_p64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_p64(arg_p64_ptr, arg_p64x2x2, 0);
	vst2q_lane_p64(arg_p64_ptr, arg_p64x2x2, 1);
	vst2q_lane_p64(arg_p64_ptr, arg_p64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_p64(arg_p64_ptr, arg_p64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_p64(arg_p64_ptr, arg_p64x1x3, 0);
	vst3_lane_p64(arg_p64_ptr, arg_p64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_p64(arg_p64_ptr, arg_p64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_p64(arg_p64_ptr, arg_p64x2x3, 0);
	vst3q_lane_p64(arg_p64_ptr, arg_p64x2x3, 1);
	vst3q_lane_p64(arg_p64_ptr, arg_p64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_p64(arg_p64_ptr, arg_p64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_p64(arg_p64_ptr, arg_p64x1x4, 0);
	vst4_lane_p64(arg_p64_ptr, arg_p64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_p64(arg_p64_ptr, arg_p64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_p64(arg_p64_ptr, arg_p64x2x4, 0);
	vst4q_lane_p64(arg_p64_ptr, arg_p64x2x4, 1);
	vst4q_lane_p64(arg_p64_ptr, arg_p64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_p64(arg_p64_ptr, arg_p64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_f16(float16x4x4_t arg_f16x4x4, float16x8_t arg_f16x8, float16x8x2_t arg_f16x8x2,
						   float16x8x3_t arg_f16x8x3, float16x4x2_t arg_f16x4x2, float16x4x3_t arg_f16x4x3,
						   float16x4_t arg_f16x4, float16_t* arg_f16_ptr, float16x8x4_t arg_f16x8x4) {
	vst1_lane_f16(arg_f16_ptr, arg_f16x4, 0);
	vst1_lane_f16(arg_f16_ptr, arg_f16x4, 3);
	vst1_lane_f16(arg_f16_ptr, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_f16(arg_f16_ptr, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_f16(arg_f16_ptr, arg_f16x8, 0);
	vst1q_lane_f16(arg_f16_ptr, arg_f16x8, 7);
	vst1q_lane_f16(arg_f16_ptr, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_f16(arg_f16_ptr, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_f16(arg_f16_ptr, arg_f16x4x2, 0);
	vst2_lane_f16(arg_f16_ptr, arg_f16x4x2, 3);
	vst2_lane_f16(arg_f16_ptr, arg_f16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_f16(arg_f16_ptr, arg_f16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_f16(arg_f16_ptr, arg_f16x8x2, 0);
	vst2q_lane_f16(arg_f16_ptr, arg_f16x8x2, 7);
	vst2q_lane_f16(arg_f16_ptr, arg_f16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_f16(arg_f16_ptr, arg_f16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_f16(arg_f16_ptr, arg_f16x4x3, 0);
	vst3_lane_f16(arg_f16_ptr, arg_f16x4x3, 3);
	vst3_lane_f16(arg_f16_ptr, arg_f16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_f16(arg_f16_ptr, arg_f16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_f16(arg_f16_ptr, arg_f16x8x3, 0);
	vst3q_lane_f16(arg_f16_ptr, arg_f16x8x3, 7);
	vst3q_lane_f16(arg_f16_ptr, arg_f16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_f16(arg_f16_ptr, arg_f16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_f16(arg_f16_ptr, arg_f16x4x4, 0);
	vst4_lane_f16(arg_f16_ptr, arg_f16x4x4, 3);
	vst4_lane_f16(arg_f16_ptr, arg_f16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_f16(arg_f16_ptr, arg_f16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_f16(arg_f16_ptr, arg_f16x8x4, 0);
	vst4q_lane_f16(arg_f16_ptr, arg_f16x8x4, 7);
	vst4q_lane_f16(arg_f16_ptr, arg_f16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_f16(arg_f16_ptr, arg_f16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_f32(float32x2x3_t arg_f32x2x3, float32x2x2_t arg_f32x2x2, float32x4_t arg_f32x4,
						   float32x4x3_t arg_f32x4x3, float32_t* arg_f32_ptr, float32x4x4_t arg_f32x4x4,
						   float32x2x4_t arg_f32x2x4, float32x2_t arg_f32x2, float32x4x2_t arg_f32x4x2) {
	vst1_lane_f32(arg_f32_ptr, arg_f32x2, 0);
	vst1_lane_f32(arg_f32_ptr, arg_f32x2, 1);
	vst1_lane_f32(arg_f32_ptr, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_f32(arg_f32_ptr, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_f32(arg_f32_ptr, arg_f32x4, 0);
	vst1q_lane_f32(arg_f32_ptr, arg_f32x4, 3);
	vst1q_lane_f32(arg_f32_ptr, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_f32(arg_f32_ptr, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_f32(arg_f32_ptr, arg_f32x2x2, 0);
	vst2_lane_f32(arg_f32_ptr, arg_f32x2x2, 1);
	vst2_lane_f32(arg_f32_ptr, arg_f32x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_f32(arg_f32_ptr, arg_f32x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_f32(arg_f32_ptr, arg_f32x4x2, 0);
	vst2q_lane_f32(arg_f32_ptr, arg_f32x4x2, 3);
	vst2q_lane_f32(arg_f32_ptr, arg_f32x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_f32(arg_f32_ptr, arg_f32x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_f32(arg_f32_ptr, arg_f32x2x3, 0);
	vst3_lane_f32(arg_f32_ptr, arg_f32x2x3, 1);
	vst3_lane_f32(arg_f32_ptr, arg_f32x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_f32(arg_f32_ptr, arg_f32x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_f32(arg_f32_ptr, arg_f32x4x3, 0);
	vst3q_lane_f32(arg_f32_ptr, arg_f32x4x3, 3);
	vst3q_lane_f32(arg_f32_ptr, arg_f32x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_f32(arg_f32_ptr, arg_f32x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_f32(arg_f32_ptr, arg_f32x2x4, 0);
	vst4_lane_f32(arg_f32_ptr, arg_f32x2x4, 1);
	vst4_lane_f32(arg_f32_ptr, arg_f32x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_f32(arg_f32_ptr, arg_f32x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_f32(arg_f32_ptr, arg_f32x4x4, 0);
	vst4q_lane_f32(arg_f32_ptr, arg_f32x4x4, 3);
	vst4q_lane_f32(arg_f32_ptr, arg_f32x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_f32(arg_f32_ptr, arg_f32x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_p8(poly8x16_t arg_p8x16, poly8x16x2_t arg_p8x16x2, poly8x8x3_t arg_p8x8x3,
						  poly8x16x3_t arg_p8x16x3, poly8x16x4_t arg_p8x16x4, poly8x8x4_t arg_p8x8x4,
						  poly8_t* arg_p8_ptr, poly8x8_t arg_p8x8, poly8x8x2_t arg_p8x8x2) {
	vst1_lane_p8(arg_p8_ptr, arg_p8x8, 0);
	vst1_lane_p8(arg_p8_ptr, arg_p8x8, 7);
	vst1_lane_p8(arg_p8_ptr, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_p8(arg_p8_ptr, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_p8(arg_p8_ptr, arg_p8x16, 0);
	vst1q_lane_p8(arg_p8_ptr, arg_p8x16, 15);
	vst1q_lane_p8(arg_p8_ptr, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_p8(arg_p8_ptr, arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_p8(arg_p8_ptr, arg_p8x8x2, 0);
	vst2_lane_p8(arg_p8_ptr, arg_p8x8x2, 7);
	vst2_lane_p8(arg_p8_ptr, arg_p8x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_p8(arg_p8_ptr, arg_p8x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_p8(arg_p8_ptr, arg_p8x8x3, 0);
	vst3_lane_p8(arg_p8_ptr, arg_p8x8x3, 7);
	vst3_lane_p8(arg_p8_ptr, arg_p8x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_p8(arg_p8_ptr, arg_p8x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_p8(arg_p8_ptr, arg_p8x8x4, 0);
	vst4_lane_p8(arg_p8_ptr, arg_p8x8x4, 7);
	vst4_lane_p8(arg_p8_ptr, arg_p8x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_p8(arg_p8_ptr, arg_p8x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_p8(arg_p8_ptr, arg_p8x16x2, 0);
	vst2q_lane_p8(arg_p8_ptr, arg_p8x16x2, 15);
	vst2q_lane_p8(arg_p8_ptr, arg_p8x16x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_p8(arg_p8_ptr, arg_p8x16x2, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_p8(arg_p8_ptr, arg_p8x16x3, 0);
	vst3q_lane_p8(arg_p8_ptr, arg_p8x16x3, 15);
	vst3q_lane_p8(arg_p8_ptr, arg_p8x16x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_p8(arg_p8_ptr, arg_p8x16x3, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_p8(arg_p8_ptr, arg_p8x16x4, 0);
	vst4q_lane_p8(arg_p8_ptr, arg_p8x16x4, 15);
	vst4q_lane_p8(arg_p8_ptr, arg_p8x16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_p8(arg_p8_ptr, arg_p8x16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_p16(poly16x4x4_t arg_p16x4x4, poly16x4_t arg_p16x4, poly16x8x2_t arg_p16x8x2,
						   poly16_t* arg_p16_ptr, poly16x8_t arg_p16x8, poly16x8x3_t arg_p16x8x3,
						   poly16x4x3_t arg_p16x4x3, poly16x8x4_t arg_p16x8x4, poly16x4x2_t arg_p16x4x2) {
	vst1_lane_p16(arg_p16_ptr, arg_p16x4, 0);
	vst1_lane_p16(arg_p16_ptr, arg_p16x4, 3);
	vst1_lane_p16(arg_p16_ptr, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_p16(arg_p16_ptr, arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_p16(arg_p16_ptr, arg_p16x8, 0);
	vst1q_lane_p16(arg_p16_ptr, arg_p16x8, 7);
	vst1q_lane_p16(arg_p16_ptr, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_p16(arg_p16_ptr, arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_p16(arg_p16_ptr, arg_p16x4x2, 0);
	vst2_lane_p16(arg_p16_ptr, arg_p16x4x2, 3);
	vst2_lane_p16(arg_p16_ptr, arg_p16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_p16(arg_p16_ptr, arg_p16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_p16(arg_p16_ptr, arg_p16x8x2, 0);
	vst2q_lane_p16(arg_p16_ptr, arg_p16x8x2, 7);
	vst2q_lane_p16(arg_p16_ptr, arg_p16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_p16(arg_p16_ptr, arg_p16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_p16(arg_p16_ptr, arg_p16x4x3, 0);
	vst3_lane_p16(arg_p16_ptr, arg_p16x4x3, 3);
	vst3_lane_p16(arg_p16_ptr, arg_p16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_p16(arg_p16_ptr, arg_p16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_p16(arg_p16_ptr, arg_p16x8x3, 0);
	vst3q_lane_p16(arg_p16_ptr, arg_p16x8x3, 7);
	vst3q_lane_p16(arg_p16_ptr, arg_p16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_p16(arg_p16_ptr, arg_p16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_p16(arg_p16_ptr, arg_p16x4x4, 0);
	vst4_lane_p16(arg_p16_ptr, arg_p16x4x4, 3);
	vst4_lane_p16(arg_p16_ptr, arg_p16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_p16(arg_p16_ptr, arg_p16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_p16(arg_p16_ptr, arg_p16x8x4, 0);
	vst4q_lane_p16(arg_p16_ptr, arg_p16x8x4, 7);
	vst4q_lane_p16(arg_p16_ptr, arg_p16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_p16(arg_p16_ptr, arg_p16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_store_f64(float64_t* arg_f64_ptr, float64x2_t arg_f64x2, float64x1x3_t arg_f64x1x3,
						   float64x2x4_t arg_f64x2x4, float64x1x4_t arg_f64x1x4, float64x1x2_t arg_f64x1x2,
						   float64x1_t arg_f64x1, float64x2x2_t arg_f64x2x2, float64x2x3_t arg_f64x2x3) {
	vst1_lane_f64(arg_f64_ptr, arg_f64x1, 0);
	vst1_lane_f64(arg_f64_ptr, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_f64(arg_f64_ptr, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_f64(arg_f64_ptr, arg_f64x2, 0);
	vst1q_lane_f64(arg_f64_ptr, arg_f64x2, 1);
	vst1q_lane_f64(arg_f64_ptr, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_f64(arg_f64_ptr, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_f64(arg_f64_ptr, arg_f64x1x2, 0);
	vst2_lane_f64(arg_f64_ptr, arg_f64x1x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_f64(arg_f64_ptr, arg_f64x1x2, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_f64(arg_f64_ptr, arg_f64x2x2, 0);
	vst2q_lane_f64(arg_f64_ptr, arg_f64x2x2, 1);
	vst2q_lane_f64(arg_f64_ptr, arg_f64x2x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_f64(arg_f64_ptr, arg_f64x2x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_f64(arg_f64_ptr, arg_f64x1x3, 0);
	vst3_lane_f64(arg_f64_ptr, arg_f64x1x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_f64(arg_f64_ptr, arg_f64x1x3, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_f64(arg_f64_ptr, arg_f64x2x3, 0);
	vst3q_lane_f64(arg_f64_ptr, arg_f64x2x3, 1);
	vst3q_lane_f64(arg_f64_ptr, arg_f64x2x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_f64(arg_f64_ptr, arg_f64x2x3, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_f64(arg_f64_ptr, arg_f64x1x4, 0);
	vst4_lane_f64(arg_f64_ptr, arg_f64x1x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_f64(arg_f64_ptr, arg_f64x1x4, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_f64(arg_f64_ptr, arg_f64x2x4, 0);
	vst4q_lane_f64(arg_f64_ptr, arg_f64x2x4, 1);
	vst4q_lane_f64(arg_f64_ptr, arg_f64x2x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_f64(arg_f64_ptr, arg_f64x2x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

