// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_vector_shift_right_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vshr_n_s8(arg_i8x8, 1);
	vshr_n_s8(arg_i8x8, 8);
	vshr_n_s8(arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_s8(arg_i8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_s8(arg_i8x16, 1);
	vshrq_n_s8(arg_i8x16, 8);
	vshrq_n_s8(arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_s8(arg_i8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vshr_n_s16(arg_i16x4, 1);
	vshr_n_s16(arg_i16x4, 16);
	vshr_n_s16(arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_s16(arg_i16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_s16(arg_i16x8, 1);
	vshrq_n_s16(arg_i16x8, 16);
	vshrq_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_s16(arg_i16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vshr_n_s32(arg_i32x2, 1);
	vshr_n_s32(arg_i32x2, 32);
	vshr_n_s32(arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_s32(arg_i32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_s32(arg_i32x4, 1);
	vshrq_n_s32(arg_i32x4, 32);
	vshrq_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_s32(arg_i32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_s64(int64_t arg_i64, int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vshr_n_s64(arg_i64x1, 1);
	vshr_n_s64(arg_i64x1, 64);
	vshr_n_s64(arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_s64(arg_i64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_s64(arg_i64x2, 1);
	vshrq_n_s64(arg_i64x2, 64);
	vshrq_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_s64(arg_i64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrd_n_s64(arg_i64, 1);
	vshrd_n_s64(arg_i64, 64);
	vshrd_n_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrd_n_s64(arg_i64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_u8(uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vshr_n_u8(arg_u8x8, 1);
	vshr_n_u8(arg_u8x8, 8);
	vshr_n_u8(arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_u8(arg_u8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_u8(arg_u8x16, 1);
	vshrq_n_u8(arg_u8x16, 8);
	vshrq_n_u8(arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_u8(arg_u8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vshr_n_u16(arg_u16x4, 1);
	vshr_n_u16(arg_u16x4, 16);
	vshr_n_u16(arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_u16(arg_u16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_u16(arg_u16x8, 1);
	vshrq_n_u16(arg_u16x8, 16);
	vshrq_n_u16(arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_u16(arg_u16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vshr_n_u32(arg_u32x2, 1);
	vshr_n_u32(arg_u32x2, 32);
	vshr_n_u32(arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_u32(arg_u32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_u32(arg_u32x4, 1);
	vshrq_n_u32(arg_u32x4, 32);
	vshrq_n_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_u32(arg_u32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_u64(uint64x2_t arg_u64x2, uint64_t arg_u64, uint64x1_t arg_u64x1) {
	vshr_n_u64(arg_u64x1, 1);
	vshr_n_u64(arg_u64x1, 64);
	vshr_n_u64(arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshr_n_u64(arg_u64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrq_n_u64(arg_u64x2, 1);
	vshrq_n_u64(arg_u64x2, 64);
	vshrq_n_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrq_n_u64(arg_u64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrd_n_u64(arg_u64, 1);
	vshrd_n_u64(arg_u64, 64);
	vshrd_n_u64(arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrd_n_u64(arg_u64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vrshr_n_s8(arg_i8x8, 1);
	vrshr_n_s8(arg_i8x8, 8);
	vrshr_n_s8(arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_s8(arg_i8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_s8(arg_i8x16, 1);
	vrshrq_n_s8(arg_i8x16, 8);
	vrshrq_n_s8(arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_s8(arg_i8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vrshr_n_s16(arg_i16x4, 1);
	vrshr_n_s16(arg_i16x4, 16);
	vrshr_n_s16(arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_s16(arg_i16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_s16(arg_i16x8, 1);
	vrshrq_n_s16(arg_i16x8, 16);
	vrshrq_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_s16(arg_i16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vrshr_n_s32(arg_i32x2, 1);
	vrshr_n_s32(arg_i32x2, 32);
	vrshr_n_s32(arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_s32(arg_i32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_s32(arg_i32x4, 1);
	vrshrq_n_s32(arg_i32x4, 32);
	vrshrq_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_s32(arg_i32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_s64(int64_t arg_i64, int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vrshr_n_s64(arg_i64x1, 1);
	vrshr_n_s64(arg_i64x1, 64);
	vrshr_n_s64(arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_s64(arg_i64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_s64(arg_i64x2, 1);
	vrshrq_n_s64(arg_i64x2, 64);
	vrshrq_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_s64(arg_i64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrd_n_s64(arg_i64, 1);
	vrshrd_n_s64(arg_i64, 64);
	vrshrd_n_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrd_n_s64(arg_i64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_u8(uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vrshr_n_u8(arg_u8x8, 1);
	vrshr_n_u8(arg_u8x8, 8);
	vrshr_n_u8(arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_u8(arg_u8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_u8(arg_u8x16, 1);
	vrshrq_n_u8(arg_u8x16, 8);
	vrshrq_n_u8(arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_u8(arg_u8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vrshr_n_u16(arg_u16x4, 1);
	vrshr_n_u16(arg_u16x4, 16);
	vrshr_n_u16(arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_u16(arg_u16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_u16(arg_u16x8, 1);
	vrshrq_n_u16(arg_u16x8, 16);
	vrshrq_n_u16(arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_u16(arg_u16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vrshr_n_u32(arg_u32x2, 1);
	vrshr_n_u32(arg_u32x2, 32);
	vrshr_n_u32(arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_u32(arg_u32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_u32(arg_u32x4, 1);
	vrshrq_n_u32(arg_u32x4, 32);
	vrshrq_n_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_u32(arg_u32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_u64(uint64x2_t arg_u64x2, uint64_t arg_u64, uint64x1_t arg_u64x1) {
	vrshr_n_u64(arg_u64x1, 1);
	vrshr_n_u64(arg_u64x1, 64);
	vrshr_n_u64(arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshr_n_u64(arg_u64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrq_n_u64(arg_u64x2, 1);
	vrshrq_n_u64(arg_u64x2, 64);
	vrshrq_n_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrq_n_u64(arg_u64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrd_n_u64(arg_u64, 1);
	vrshrd_n_u64(arg_u64, 64);
	vrshrd_n_u64(arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrd_n_u64(arg_u64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vsra_n_s8(arg_i8x8, arg_i8x8, 1);
	vsra_n_s8(arg_i8x8, arg_i8x8, 8);
	vsra_n_s8(arg_i8x8, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_s8(arg_i8x8, arg_i8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_s8(arg_i8x16, arg_i8x16, 1);
	vsraq_n_s8(arg_i8x16, arg_i8x16, 8);
	vsraq_n_s8(arg_i8x16, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_s8(arg_i8x16, arg_i8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vsra_n_s16(arg_i16x4, arg_i16x4, 1);
	vsra_n_s16(arg_i16x4, arg_i16x4, 16);
	vsra_n_s16(arg_i16x4, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_s16(arg_i16x4, arg_i16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_s16(arg_i16x8, arg_i16x8, 1);
	vsraq_n_s16(arg_i16x8, arg_i16x8, 16);
	vsraq_n_s16(arg_i16x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_s16(arg_i16x8, arg_i16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vsra_n_s32(arg_i32x2, arg_i32x2, 1);
	vsra_n_s32(arg_i32x2, arg_i32x2, 32);
	vsra_n_s32(arg_i32x2, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_s32(arg_i32x2, arg_i32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_s32(arg_i32x4, arg_i32x4, 1);
	vsraq_n_s32(arg_i32x4, arg_i32x4, 32);
	vsraq_n_s32(arg_i32x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_s32(arg_i32x4, arg_i32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_s64(int64_t arg_i64, int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vsra_n_s64(arg_i64x1, arg_i64x1, 1);
	vsra_n_s64(arg_i64x1, arg_i64x1, 64);
	vsra_n_s64(arg_i64x1, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_s64(arg_i64x1, arg_i64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_s64(arg_i64x2, arg_i64x2, 1);
	vsraq_n_s64(arg_i64x2, arg_i64x2, 64);
	vsraq_n_s64(arg_i64x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_s64(arg_i64x2, arg_i64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsrad_n_s64(arg_i64, arg_i64, 1);
	vsrad_n_s64(arg_i64, arg_i64, 64);
	vsrad_n_s64(arg_i64, arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsrad_n_s64(arg_i64, arg_i64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_u8(uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vsra_n_u8(arg_u8x8, arg_u8x8, 1);
	vsra_n_u8(arg_u8x8, arg_u8x8, 8);
	vsra_n_u8(arg_u8x8, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_u8(arg_u8x8, arg_u8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_u8(arg_u8x16, arg_u8x16, 1);
	vsraq_n_u8(arg_u8x16, arg_u8x16, 8);
	vsraq_n_u8(arg_u8x16, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_u8(arg_u8x16, arg_u8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vsra_n_u16(arg_u16x4, arg_u16x4, 1);
	vsra_n_u16(arg_u16x4, arg_u16x4, 16);
	vsra_n_u16(arg_u16x4, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_u16(arg_u16x4, arg_u16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_u16(arg_u16x8, arg_u16x8, 1);
	vsraq_n_u16(arg_u16x8, arg_u16x8, 16);
	vsraq_n_u16(arg_u16x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_u16(arg_u16x8, arg_u16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vsra_n_u32(arg_u32x2, arg_u32x2, 1);
	vsra_n_u32(arg_u32x2, arg_u32x2, 32);
	vsra_n_u32(arg_u32x2, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_u32(arg_u32x2, arg_u32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_u32(arg_u32x4, arg_u32x4, 1);
	vsraq_n_u32(arg_u32x4, arg_u32x4, 32);
	vsraq_n_u32(arg_u32x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_u32(arg_u32x4, arg_u32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_accumulate_u64(uint64x2_t arg_u64x2, uint64_t arg_u64, uint64x1_t arg_u64x1) {
	vsra_n_u64(arg_u64x1, arg_u64x1, 1);
	vsra_n_u64(arg_u64x1, arg_u64x1, 64);
	vsra_n_u64(arg_u64x1, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsra_n_u64(arg_u64x1, arg_u64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsraq_n_u64(arg_u64x2, arg_u64x2, 1);
	vsraq_n_u64(arg_u64x2, arg_u64x2, 64);
	vsraq_n_u64(arg_u64x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsraq_n_u64(arg_u64x2, arg_u64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsrad_n_u64(arg_u64, arg_u64, 1);
	vsrad_n_u64(arg_u64, arg_u64, 64);
	vsrad_n_u64(arg_u64, arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsrad_n_u64(arg_u64, arg_u64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vrsra_n_s8(arg_i8x8, arg_i8x8, 1);
	vrsra_n_s8(arg_i8x8, arg_i8x8, 8);
	vrsra_n_s8(arg_i8x8, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_s8(arg_i8x8, arg_i8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_s8(arg_i8x16, arg_i8x16, 1);
	vrsraq_n_s8(arg_i8x16, arg_i8x16, 8);
	vrsraq_n_s8(arg_i8x16, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_s8(arg_i8x16, arg_i8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vrsra_n_s16(arg_i16x4, arg_i16x4, 1);
	vrsra_n_s16(arg_i16x4, arg_i16x4, 16);
	vrsra_n_s16(arg_i16x4, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_s16(arg_i16x4, arg_i16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_s16(arg_i16x8, arg_i16x8, 1);
	vrsraq_n_s16(arg_i16x8, arg_i16x8, 16);
	vrsraq_n_s16(arg_i16x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_s16(arg_i16x8, arg_i16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vrsra_n_s32(arg_i32x2, arg_i32x2, 1);
	vrsra_n_s32(arg_i32x2, arg_i32x2, 32);
	vrsra_n_s32(arg_i32x2, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_s32(arg_i32x2, arg_i32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_s32(arg_i32x4, arg_i32x4, 1);
	vrsraq_n_s32(arg_i32x4, arg_i32x4, 32);
	vrsraq_n_s32(arg_i32x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_s32(arg_i32x4, arg_i32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_s64(int64_t arg_i64, int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vrsra_n_s64(arg_i64x1, arg_i64x1, 1);
	vrsra_n_s64(arg_i64x1, arg_i64x1, 64);
	vrsra_n_s64(arg_i64x1, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_s64(arg_i64x1, arg_i64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_s64(arg_i64x2, arg_i64x2, 1);
	vrsraq_n_s64(arg_i64x2, arg_i64x2, 64);
	vrsraq_n_s64(arg_i64x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_s64(arg_i64x2, arg_i64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsrad_n_s64(arg_i64, arg_i64, 1);
	vrsrad_n_s64(arg_i64, arg_i64, 64);
	vrsrad_n_s64(arg_i64, arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsrad_n_s64(arg_i64, arg_i64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_u8(uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vrsra_n_u8(arg_u8x8, arg_u8x8, 1);
	vrsra_n_u8(arg_u8x8, arg_u8x8, 8);
	vrsra_n_u8(arg_u8x8, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_u8(arg_u8x8, arg_u8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_u8(arg_u8x16, arg_u8x16, 1);
	vrsraq_n_u8(arg_u8x16, arg_u8x16, 8);
	vrsraq_n_u8(arg_u8x16, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_u8(arg_u8x16, arg_u8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vrsra_n_u16(arg_u16x4, arg_u16x4, 1);
	vrsra_n_u16(arg_u16x4, arg_u16x4, 16);
	vrsra_n_u16(arg_u16x4, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_u16(arg_u16x4, arg_u16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_u16(arg_u16x8, arg_u16x8, 1);
	vrsraq_n_u16(arg_u16x8, arg_u16x8, 16);
	vrsraq_n_u16(arg_u16x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_u16(arg_u16x8, arg_u16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vrsra_n_u32(arg_u32x2, arg_u32x2, 1);
	vrsra_n_u32(arg_u32x2, arg_u32x2, 32);
	vrsra_n_u32(arg_u32x2, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_u32(arg_u32x2, arg_u32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_u32(arg_u32x4, arg_u32x4, 1);
	vrsraq_n_u32(arg_u32x4, arg_u32x4, 32);
	vrsraq_n_u32(arg_u32x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_u32(arg_u32x4, arg_u32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_accumulate_u64(uint64x2_t arg_u64x2, uint64_t arg_u64, uint64x1_t arg_u64x1) {
	vrsra_n_u64(arg_u64x1, arg_u64x1, 1);
	vrsra_n_u64(arg_u64x1, arg_u64x1, 64);
	vrsra_n_u64(arg_u64x1, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsra_n_u64(arg_u64x1, arg_u64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsraq_n_u64(arg_u64x2, arg_u64x2, 1);
	vrsraq_n_u64(arg_u64x2, arg_u64x2, 64);
	vrsraq_n_u64(arg_u64x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsraq_n_u64(arg_u64x2, arg_u64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrsrad_n_u64(arg_u64, arg_u64, 1);
	vrsrad_n_u64(arg_u64, arg_u64, 64);
	vrsrad_n_u64(arg_u64, arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrsrad_n_u64(arg_u64, arg_u64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_narrow_s16(int16x8_t arg_i16x8, int8x8_t arg_i8x8) {
	vshrn_n_s16(arg_i16x8, 1);
	vshrn_n_s16(arg_i16x8, 8);
	vshrn_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_n_s16(arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrn_high_n_s16(arg_i8x8, arg_i16x8, 1);
	vshrn_high_n_s16(arg_i8x8, arg_i16x8, 8);
	vshrn_high_n_s16(arg_i8x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_high_n_s16(arg_i8x8, arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_narrow_s32(int32x4_t arg_i32x4, int16x4_t arg_i16x4) {
	vshrn_n_s32(arg_i32x4, 1);
	vshrn_n_s32(arg_i32x4, 16);
	vshrn_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_n_s32(arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrn_high_n_s32(arg_i16x4, arg_i32x4, 1);
	vshrn_high_n_s32(arg_i16x4, arg_i32x4, 16);
	vshrn_high_n_s32(arg_i16x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_high_n_s32(arg_i16x4, arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_narrow_s64(int32x2_t arg_i32x2, int64x2_t arg_i64x2) {
	vshrn_n_s64(arg_i64x2, 1);
	vshrn_n_s64(arg_i64x2, 32);
	vshrn_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_n_s64(arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrn_high_n_s64(arg_i32x2, arg_i64x2, 1);
	vshrn_high_n_s64(arg_i32x2, arg_i64x2, 32);
	vshrn_high_n_s64(arg_i32x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_high_n_s64(arg_i32x2, arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_narrow_u16(uint16x8_t arg_u16x8, uint8x8_t arg_u8x8) {
	vshrn_n_u16(arg_u16x8, 1);
	vshrn_n_u16(arg_u16x8, 8);
	vshrn_n_u16(arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_n_u16(arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrn_high_n_u16(arg_u8x8, arg_u16x8, 1);
	vshrn_high_n_u16(arg_u8x8, arg_u16x8, 8);
	vshrn_high_n_u16(arg_u8x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_high_n_u16(arg_u8x8, arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_narrow_u32(uint32x4_t arg_u32x4, uint16x4_t arg_u16x4) {
	vshrn_n_u32(arg_u32x4, 1);
	vshrn_n_u32(arg_u32x4, 16);
	vshrn_n_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_n_u32(arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrn_high_n_u32(arg_u16x4, arg_u32x4, 1);
	vshrn_high_n_u32(arg_u16x4, arg_u32x4, 16);
	vshrn_high_n_u32(arg_u16x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_high_n_u32(arg_u16x4, arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_narrow_u64(uint64x2_t arg_u64x2, uint32x2_t arg_u32x2) {
	vshrn_n_u64(arg_u64x2, 1);
	vshrn_n_u64(arg_u64x2, 32);
	vshrn_n_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_n_u64(arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshrn_high_n_u64(arg_u32x2, arg_u64x2, 1);
	vshrn_high_n_u64(arg_u32x2, arg_u64x2, 32);
	vshrn_high_n_u64(arg_u32x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshrn_high_n_u64(arg_u32x2, arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_right_and_narrow_s16(int16x8_t arg_i16x8, uint8x8_t arg_u8x8, int16_t arg_i16, int8x8_t arg_i8x8) {
	vqshrun_n_s16(arg_i16x8, 1);
	vqshrun_n_s16(arg_i16x8, 8);
	vqshrun_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrun_n_s16(arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrunh_n_s16(arg_i16, 1);
	vqshrunh_n_s16(arg_i16, 8);
	vqshrunh_n_s16(arg_i16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrunh_n_s16(arg_i16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrun_high_n_s16(arg_u8x8, arg_i16x8, 1);
	vqshrun_high_n_s16(arg_u8x8, arg_i16x8, 8);
	vqshrun_high_n_s16(arg_u8x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrun_high_n_s16(arg_u8x8, arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_n_s16(arg_i16x8, 1);
	vqshrn_n_s16(arg_i16x8, 8);
	vqshrn_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_n_s16(arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrnh_n_s16(arg_i16, 1);
	vqshrnh_n_s16(arg_i16, 8);
	vqshrnh_n_s16(arg_i16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrnh_n_s16(arg_i16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_high_n_s16(arg_i8x8, arg_i16x8, 1);
	vqshrn_high_n_s16(arg_i8x8, arg_i16x8, 8);
	vqshrn_high_n_s16(arg_i8x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_high_n_s16(arg_i8x8, arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_right_and_narrow_s32(int16x4_t arg_i16x4, int32_t arg_i32, int32x4_t arg_i32x4, uint16x4_t arg_u16x4) {
	vqshrun_n_s32(arg_i32x4, 1);
	vqshrun_n_s32(arg_i32x4, 16);
	vqshrun_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrun_n_s32(arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshruns_n_s32(arg_i32, 1);
	vqshruns_n_s32(arg_i32, 16);
	vqshruns_n_s32(arg_i32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshruns_n_s32(arg_i32, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrun_high_n_s32(arg_u16x4, arg_i32x4, 1);
	vqshrun_high_n_s32(arg_u16x4, arg_i32x4, 16);
	vqshrun_high_n_s32(arg_u16x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrun_high_n_s32(arg_u16x4, arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_n_s32(arg_i32x4, 1);
	vqshrn_n_s32(arg_i32x4, 16);
	vqshrn_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_n_s32(arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrns_n_s32(arg_i32, 1);
	vqshrns_n_s32(arg_i32, 16);
	vqshrns_n_s32(arg_i32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrns_n_s32(arg_i32, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_high_n_s32(arg_i16x4, arg_i32x4, 1);
	vqshrn_high_n_s32(arg_i16x4, arg_i32x4, 16);
	vqshrn_high_n_s32(arg_i16x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_high_n_s32(arg_i16x4, arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_right_and_narrow_s64(uint32x2_t arg_u32x2, int64x2_t arg_i64x2, int32x2_t arg_i32x2, int64_t arg_i64) {
	vqshrun_n_s64(arg_i64x2, 1);
	vqshrun_n_s64(arg_i64x2, 32);
	vqshrun_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrun_n_s64(arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrund_n_s64(arg_i64, 1);
	vqshrund_n_s64(arg_i64, 32);
	vqshrund_n_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrund_n_s64(arg_i64, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrun_high_n_s64(arg_u32x2, arg_i64x2, 1);
	vqshrun_high_n_s64(arg_u32x2, arg_i64x2, 32);
	vqshrun_high_n_s64(arg_u32x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrun_high_n_s64(arg_u32x2, arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_n_s64(arg_i64x2, 1);
	vqshrn_n_s64(arg_i64x2, 32);
	vqshrn_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_n_s64(arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrnd_n_s64(arg_i64, 1);
	vqshrnd_n_s64(arg_i64, 32);
	vqshrnd_n_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrnd_n_s64(arg_i64, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_high_n_s64(arg_i32x2, arg_i64x2, 1);
	vqshrn_high_n_s64(arg_i32x2, arg_i64x2, 32);
	vqshrn_high_n_s64(arg_i32x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_high_n_s64(arg_i32x2, arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_right_and_narrow_u16(uint16x8_t arg_u16x8, uint16_t arg_u16, uint8x8_t arg_u8x8) {
	vqshrn_n_u16(arg_u16x8, 1);
	vqshrn_n_u16(arg_u16x8, 8);
	vqshrn_n_u16(arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_n_u16(arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrnh_n_u16(arg_u16, 1);
	vqshrnh_n_u16(arg_u16, 8);
	vqshrnh_n_u16(arg_u16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrnh_n_u16(arg_u16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_high_n_u16(arg_u8x8, arg_u16x8, 1);
	vqshrn_high_n_u16(arg_u8x8, arg_u16x8, 8);
	vqshrn_high_n_u16(arg_u8x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_high_n_u16(arg_u8x8, arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_right_and_narrow_u32(uint32x4_t arg_u32x4, uint32_t arg_u32, uint16x4_t arg_u16x4) {
	vqshrn_n_u32(arg_u32x4, 1);
	vqshrn_n_u32(arg_u32x4, 16);
	vqshrn_n_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_n_u32(arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrns_n_u32(arg_u32, 1);
	vqshrns_n_u32(arg_u32, 16);
	vqshrns_n_u32(arg_u32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrns_n_u32(arg_u32, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_high_n_u32(arg_u16x4, arg_u32x4, 1);
	vqshrn_high_n_u32(arg_u16x4, arg_u32x4, 16);
	vqshrn_high_n_u32(arg_u16x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_high_n_u32(arg_u16x4, arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_right_and_narrow_u64(uint64x2_t arg_u64x2, uint32x2_t arg_u32x2, uint64_t arg_u64) {
	vqshrn_n_u64(arg_u64x2, 1);
	vqshrn_n_u64(arg_u64x2, 32);
	vqshrn_n_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_n_u64(arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrnd_n_u64(arg_u64, 1);
	vqshrnd_n_u64(arg_u64, 32);
	vqshrnd_n_u64(arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrnd_n_u64(arg_u64, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshrn_high_n_u64(arg_u32x2, arg_u64x2, 1);
	vqshrn_high_n_u64(arg_u32x2, arg_u64x2, 32);
	vqshrn_high_n_u64(arg_u32x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshrn_high_n_u64(arg_u32x2, arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_rounding_shift_right_and_narrow_s16(int16x8_t arg_i16x8, uint8x8_t arg_u8x8,
																int16_t arg_i16, int8x8_t arg_i8x8) {
	vqrshrun_n_s16(arg_i16x8, 1);
	vqrshrun_n_s16(arg_i16x8, 8);
	vqrshrun_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrun_n_s16(arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrunh_n_s16(arg_i16, 1);
	vqrshrunh_n_s16(arg_i16, 8);
	vqrshrunh_n_s16(arg_i16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrunh_n_s16(arg_i16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrun_high_n_s16(arg_u8x8, arg_i16x8, 1);
	vqrshrun_high_n_s16(arg_u8x8, arg_i16x8, 8);
	vqrshrun_high_n_s16(arg_u8x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrun_high_n_s16(arg_u8x8, arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_n_s16(arg_i16x8, 1);
	vqrshrn_n_s16(arg_i16x8, 8);
	vqrshrn_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_n_s16(arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrnh_n_s16(arg_i16, 1);
	vqrshrnh_n_s16(arg_i16, 8);
	vqrshrnh_n_s16(arg_i16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrnh_n_s16(arg_i16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_high_n_s16(arg_i8x8, arg_i16x8, 1);
	vqrshrn_high_n_s16(arg_i8x8, arg_i16x8, 8);
	vqrshrn_high_n_s16(arg_i8x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_high_n_s16(arg_i8x8, arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_rounding_shift_right_and_narrow_s32(int16x4_t arg_i16x4, int32_t arg_i32,
																 int32x4_t arg_i32x4, uint16x4_t arg_u16x4) {
	vqrshrun_n_s32(arg_i32x4, 1);
	vqrshrun_n_s32(arg_i32x4, 16);
	vqrshrun_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrun_n_s32(arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshruns_n_s32(arg_i32, 1);
	vqrshruns_n_s32(arg_i32, 16);
	vqrshruns_n_s32(arg_i32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshruns_n_s32(arg_i32, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrun_high_n_s32(arg_u16x4, arg_i32x4, 1);
	vqrshrun_high_n_s32(arg_u16x4, arg_i32x4, 16);
	vqrshrun_high_n_s32(arg_u16x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrun_high_n_s32(arg_u16x4, arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_n_s32(arg_i32x4, 1);
	vqrshrn_n_s32(arg_i32x4, 16);
	vqrshrn_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_n_s32(arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrns_n_s32(arg_i32, 1);
	vqrshrns_n_s32(arg_i32, 16);
	vqrshrns_n_s32(arg_i32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrns_n_s32(arg_i32, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_high_n_s32(arg_i16x4, arg_i32x4, 1);
	vqrshrn_high_n_s32(arg_i16x4, arg_i32x4, 16);
	vqrshrn_high_n_s32(arg_i16x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_high_n_s32(arg_i16x4, arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_rounding_shift_right_and_narrow_s64(uint32x2_t arg_u32x2, int64x2_t arg_i64x2,
																int32x2_t arg_i32x2, int64_t arg_i64) {
	vqrshrun_n_s64(arg_i64x2, 1);
	vqrshrun_n_s64(arg_i64x2, 32);
	vqrshrun_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrun_n_s64(arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrund_n_s64(arg_i64, 1);
	vqrshrund_n_s64(arg_i64, 32);
	vqrshrund_n_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrund_n_s64(arg_i64, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrun_high_n_s64(arg_u32x2, arg_i64x2, 1);
	vqrshrun_high_n_s64(arg_u32x2, arg_i64x2, 32);
	vqrshrun_high_n_s64(arg_u32x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrun_high_n_s64(arg_u32x2, arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_n_s64(arg_i64x2, 1);
	vqrshrn_n_s64(arg_i64x2, 32);
	vqrshrn_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_n_s64(arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrnd_n_s64(arg_i64, 1);
	vqrshrnd_n_s64(arg_i64, 32);
	vqrshrnd_n_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrnd_n_s64(arg_i64, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_high_n_s64(arg_i32x2, arg_i64x2, 1);
	vqrshrn_high_n_s64(arg_i32x2, arg_i64x2, 32);
	vqrshrn_high_n_s64(arg_i32x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_high_n_s64(arg_i32x2, arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_rounding_shift_right_and_narrow_u16(uint16x8_t arg_u16x8, uint16_t arg_u16,
																uint8x8_t arg_u8x8) {
	vqrshrn_n_u16(arg_u16x8, 1);
	vqrshrn_n_u16(arg_u16x8, 8);
	vqrshrn_n_u16(arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_n_u16(arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrnh_n_u16(arg_u16, 1);
	vqrshrnh_n_u16(arg_u16, 8);
	vqrshrnh_n_u16(arg_u16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrnh_n_u16(arg_u16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_high_n_u16(arg_u8x8, arg_u16x8, 1);
	vqrshrn_high_n_u16(arg_u8x8, arg_u16x8, 8);
	vqrshrn_high_n_u16(arg_u8x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_high_n_u16(arg_u8x8, arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_rounding_shift_right_and_narrow_u32(uint32x4_t arg_u32x4, uint32_t arg_u32,
																uint16x4_t arg_u16x4) {
	vqrshrn_n_u32(arg_u32x4, 1);
	vqrshrn_n_u32(arg_u32x4, 16);
	vqrshrn_n_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_n_u32(arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrns_n_u32(arg_u32, 1);
	vqrshrns_n_u32(arg_u32, 16);
	vqrshrns_n_u32(arg_u32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrns_n_u32(arg_u32, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_high_n_u32(arg_u16x4, arg_u32x4, 1);
	vqrshrn_high_n_u32(arg_u16x4, arg_u32x4, 16);
	vqrshrn_high_n_u32(arg_u16x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_high_n_u32(arg_u16x4, arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_rounding_shift_right_and_narrow_u64(uint64x2_t arg_u64x2, uint32x2_t arg_u32x2,
																uint64_t arg_u64) {
	vqrshrn_n_u64(arg_u64x2, 1);
	vqrshrn_n_u64(arg_u64x2, 32);
	vqrshrn_n_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_n_u64(arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrnd_n_u64(arg_u64, 1);
	vqrshrnd_n_u64(arg_u64, 32);
	vqrshrnd_n_u64(arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrnd_n_u64(arg_u64, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrshrn_high_n_u64(arg_u32x2, arg_u64x2, 1);
	vqrshrn_high_n_u64(arg_u32x2, arg_u64x2, 32);
	vqrshrn_high_n_u64(arg_u32x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrshrn_high_n_u64(arg_u32x2, arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_narrow_s16(int16x8_t arg_i16x8, int8x8_t arg_i8x8) {
	vrshrn_n_s16(arg_i16x8, 1);
	vrshrn_n_s16(arg_i16x8, 8);
	vrshrn_n_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_n_s16(arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrn_high_n_s16(arg_i8x8, arg_i16x8, 1);
	vrshrn_high_n_s16(arg_i8x8, arg_i16x8, 8);
	vrshrn_high_n_s16(arg_i8x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_high_n_s16(arg_i8x8, arg_i16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_narrow_s32(int32x4_t arg_i32x4, int16x4_t arg_i16x4) {
	vrshrn_n_s32(arg_i32x4, 1);
	vrshrn_n_s32(arg_i32x4, 16);
	vrshrn_n_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_n_s32(arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrn_high_n_s32(arg_i16x4, arg_i32x4, 1);
	vrshrn_high_n_s32(arg_i16x4, arg_i32x4, 16);
	vrshrn_high_n_s32(arg_i16x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_high_n_s32(arg_i16x4, arg_i32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_narrow_s64(int32x2_t arg_i32x2, int64x2_t arg_i64x2) {
	vrshrn_n_s64(arg_i64x2, 1);
	vrshrn_n_s64(arg_i64x2, 32);
	vrshrn_n_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_n_s64(arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrn_high_n_s64(arg_i32x2, arg_i64x2, 1);
	vrshrn_high_n_s64(arg_i32x2, arg_i64x2, 32);
	vrshrn_high_n_s64(arg_i32x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_high_n_s64(arg_i32x2, arg_i64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_narrow_u16(uint16x8_t arg_u16x8, uint8x8_t arg_u8x8) {
	vrshrn_n_u16(arg_u16x8, 1);
	vrshrn_n_u16(arg_u16x8, 8);
	vrshrn_n_u16(arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_n_u16(arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrn_high_n_u16(arg_u8x8, arg_u16x8, 1);
	vrshrn_high_n_u16(arg_u8x8, arg_u16x8, 8);
	vrshrn_high_n_u16(arg_u8x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_high_n_u16(arg_u8x8, arg_u16x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_narrow_u32(uint32x4_t arg_u32x4, uint16x4_t arg_u16x4) {
	vrshrn_n_u32(arg_u32x4, 1);
	vrshrn_n_u32(arg_u32x4, 16);
	vrshrn_n_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_n_u32(arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrn_high_n_u32(arg_u16x4, arg_u32x4, 1);
	vrshrn_high_n_u32(arg_u16x4, arg_u32x4, 16);
	vrshrn_high_n_u32(arg_u16x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_high_n_u32(arg_u16x4, arg_u32x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_rounding_shift_right_and_narrow_u64(uint64x2_t arg_u64x2, uint32x2_t arg_u32x2) {
	vrshrn_n_u64(arg_u64x2, 1);
	vrshrn_n_u64(arg_u64x2, 32);
	vrshrn_n_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_n_u64(arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vrshrn_high_n_u64(arg_u32x2, arg_u64x2, 1);
	vrshrn_high_n_u64(arg_u32x2, arg_u64x2, 32);
	vrshrn_high_n_u64(arg_u32x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vrshrn_high_n_u64(arg_u32x2, arg_u64x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vsri_n_s8(arg_i8x8, arg_i8x8, 1);
	vsri_n_s8(arg_i8x8, arg_i8x8, 8);
	vsri_n_s8(arg_i8x8, arg_i8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_s8(arg_i8x8, arg_i8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_s8(arg_i8x16, arg_i8x16, 1);
	vsriq_n_s8(arg_i8x16, arg_i8x16, 8);
	vsriq_n_s8(arg_i8x16, arg_i8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_s8(arg_i8x16, arg_i8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vsri_n_s16(arg_i16x4, arg_i16x4, 1);
	vsri_n_s16(arg_i16x4, arg_i16x4, 16);
	vsri_n_s16(arg_i16x4, arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_s16(arg_i16x4, arg_i16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_s16(arg_i16x8, arg_i16x8, 1);
	vsriq_n_s16(arg_i16x8, arg_i16x8, 16);
	vsriq_n_s16(arg_i16x8, arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_s16(arg_i16x8, arg_i16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vsri_n_s32(arg_i32x2, arg_i32x2, 1);
	vsri_n_s32(arg_i32x2, arg_i32x2, 32);
	vsri_n_s32(arg_i32x2, arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_s32(arg_i32x2, arg_i32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_s32(arg_i32x4, arg_i32x4, 1);
	vsriq_n_s32(arg_i32x4, arg_i32x4, 32);
	vsriq_n_s32(arg_i32x4, arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_s32(arg_i32x4, arg_i32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_s64(int64_t arg_i64, int64x1_t arg_i64x1, int64x2_t arg_i64x2) {
	vsri_n_s64(arg_i64x1, arg_i64x1, 1);
	vsri_n_s64(arg_i64x1, arg_i64x1, 64);
	vsri_n_s64(arg_i64x1, arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_s64(arg_i64x1, arg_i64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_s64(arg_i64x2, arg_i64x2, 1);
	vsriq_n_s64(arg_i64x2, arg_i64x2, 64);
	vsriq_n_s64(arg_i64x2, arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_s64(arg_i64x2, arg_i64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsrid_n_s64(arg_i64, arg_i64, 1);
	vsrid_n_s64(arg_i64, arg_i64, 64);
	vsrid_n_s64(arg_i64, arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsrid_n_s64(arg_i64, arg_i64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_u8(uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vsri_n_u8(arg_u8x8, arg_u8x8, 1);
	vsri_n_u8(arg_u8x8, arg_u8x8, 8);
	vsri_n_u8(arg_u8x8, arg_u8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_u8(arg_u8x8, arg_u8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_u8(arg_u8x16, arg_u8x16, 1);
	vsriq_n_u8(arg_u8x16, arg_u8x16, 8);
	vsriq_n_u8(arg_u8x16, arg_u8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_u8(arg_u8x16, arg_u8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vsri_n_u16(arg_u16x4, arg_u16x4, 1);
	vsri_n_u16(arg_u16x4, arg_u16x4, 16);
	vsri_n_u16(arg_u16x4, arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_u16(arg_u16x4, arg_u16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_u16(arg_u16x8, arg_u16x8, 1);
	vsriq_n_u16(arg_u16x8, arg_u16x8, 16);
	vsriq_n_u16(arg_u16x8, arg_u16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_u16(arg_u16x8, arg_u16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vsri_n_u32(arg_u32x2, arg_u32x2, 1);
	vsri_n_u32(arg_u32x2, arg_u32x2, 32);
	vsri_n_u32(arg_u32x2, arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_u32(arg_u32x2, arg_u32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_u32(arg_u32x4, arg_u32x4, 1);
	vsriq_n_u32(arg_u32x4, arg_u32x4, 32);
	vsriq_n_u32(arg_u32x4, arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_u32(arg_u32x4, arg_u32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_u64(uint64x2_t arg_u64x2, uint64_t arg_u64, uint64x1_t arg_u64x1) {
	vsri_n_u64(arg_u64x1, arg_u64x1, 1);
	vsri_n_u64(arg_u64x1, arg_u64x1, 64);
	vsri_n_u64(arg_u64x1, arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_u64(arg_u64x1, arg_u64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_u64(arg_u64x2, arg_u64x2, 1);
	vsriq_n_u64(arg_u64x2, arg_u64x2, 64);
	vsriq_n_u64(arg_u64x2, arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_u64(arg_u64x2, arg_u64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsrid_n_u64(arg_u64, arg_u64, 1);
	vsrid_n_u64(arg_u64, arg_u64, 64);
	vsrid_n_u64(arg_u64, arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsrid_n_u64(arg_u64, arg_u64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_p64(poly64x2_t arg_p64x2, poly64x1_t arg_p64x1) {
	vsri_n_p64(arg_p64x1, arg_p64x1, 1);
	vsri_n_p64(arg_p64x1, arg_p64x1, 64);
	vsri_n_p64(arg_p64x1, arg_p64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_p64(arg_p64x1, arg_p64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_p64(arg_p64x2, arg_p64x2, 1);
	vsriq_n_p64(arg_p64x2, arg_p64x2, 64);
	vsriq_n_p64(arg_p64x2, arg_p64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_p64(arg_p64x2, arg_p64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_p8(poly8x16_t arg_p8x16, poly8x8_t arg_p8x8) {
	vsri_n_p8(arg_p8x8, arg_p8x8, 1);
	vsri_n_p8(arg_p8x8, arg_p8x8, 8);
	vsri_n_p8(arg_p8x8, arg_p8x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_p8(arg_p8x8, arg_p8x8, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_p8(arg_p8x16, arg_p8x16, 1);
	vsriq_n_p8(arg_p8x16, arg_p8x16, 8);
	vsriq_n_p8(arg_p8x16, arg_p8x16, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_p8(arg_p8x16, arg_p8x16, 9); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_right_and_insert_p16(poly16x4_t arg_p16x4, poly16x8_t arg_p16x8) {
	vsri_n_p16(arg_p16x4, arg_p16x4, 1);
	vsri_n_p16(arg_p16x4, arg_p16x4, 16);
	vsri_n_p16(arg_p16x4, arg_p16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsri_n_p16(arg_p16x4, arg_p16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsriq_n_p16(arg_p16x8, arg_p16x8, 1);
	vsriq_n_p16(arg_p16x8, arg_p16x8, 16);
	vsriq_n_p16(arg_p16x8, arg_p16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsriq_n_p16(arg_p16x8, arg_p16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

