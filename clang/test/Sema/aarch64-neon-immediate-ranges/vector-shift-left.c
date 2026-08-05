// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

// Widening left-shifts should have a range of 0..(sizeinbits(arg)-1), this range has had
// to be weakened to 0..((sizeinbits(arg)*2)-1) due to a use of vshll_n_s16 with an
// out-of-bounds immediate in the defintiion of vcvt_f32_bf16. As a result, the upper bounds
// of widening left-shift intrinsics are not currently tested here.

void test_vector_shift_left_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vshl_n_s8(arg_i8x8, 0);
	vshl_n_s8(arg_i8x8, 7);
	vshl_n_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_s8(arg_i8x16, 0);
	vshlq_n_s8(arg_i8x16, 7);
	vshlq_n_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_s8(arg_i8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vshl_n_s16(arg_i16x4, 0);
	vshl_n_s16(arg_i16x4, 15);
	vshl_n_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_s16(arg_i16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_s16(arg_i16x8, 0);
	vshlq_n_s16(arg_i16x8, 15);
	vshlq_n_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_s16(arg_i16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vshl_n_s32(arg_i32x2, 0);
	vshl_n_s32(arg_i32x2, 31);
	vshl_n_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_s32(arg_i32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_s32(arg_i32x4, 0);
	vshlq_n_s32(arg_i32x4, 31);
	vshlq_n_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_s32(arg_i32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_s64(int64_t arg_i64, int64x2_t arg_i64x2, int64x1_t arg_i64x1) {
	vshl_n_s64(arg_i64x1, 0);
	vshl_n_s64(arg_i64x1, 63);
	vshl_n_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_s64(arg_i64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_s64(arg_i64x2, 0);
	vshlq_n_s64(arg_i64x2, 63);
	vshlq_n_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_s64(arg_i64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshld_n_s64(arg_i64, 0);
	vshld_n_s64(arg_i64, 63);
	vshld_n_s64(arg_i64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshld_n_s64(arg_i64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vshl_n_u8(arg_u8x8, 0);
	vshl_n_u8(arg_u8x8, 7);
	vshl_n_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_u8(arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_u8(arg_u8x16, 0);
	vshlq_n_u8(arg_u8x16, 7);
	vshlq_n_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_u8(arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vshl_n_u16(arg_u16x4, 0);
	vshl_n_u16(arg_u16x4, 15);
	vshl_n_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_u16(arg_u16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_u16(arg_u16x8, 0);
	vshlq_n_u16(arg_u16x8, 15);
	vshlq_n_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_u16(arg_u16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vshl_n_u32(arg_u32x2, 0);
	vshl_n_u32(arg_u32x2, 31);
	vshl_n_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_u32(arg_u32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_u32(arg_u32x4, 0);
	vshlq_n_u32(arg_u32x4, 31);
	vshlq_n_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_u32(arg_u32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_u64(uint64x1_t arg_u64x1, uint64_t arg_u64, uint64x2_t arg_u64x2) {
	vshl_n_u64(arg_u64x1, 0);
	vshl_n_u64(arg_u64x1, 63);
	vshl_n_u64(arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshl_n_u64(arg_u64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshlq_n_u64(arg_u64x2, 0);
	vshlq_n_u64(arg_u64x2, 63);
	vshlq_n_u64(arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshlq_n_u64(arg_u64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshld_n_u64(arg_u64, 0);
	vshld_n_u64(arg_u64, 63);
	vshld_n_u64(arg_u64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vshld_n_u64(arg_u64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16, int8_t arg_i8) {
	vqshl_n_s8(arg_i8x8, 0);
	vqshl_n_s8(arg_i8x8, 7);
	vqshl_n_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_s8(arg_i8x16, 0);
	vqshlq_n_s8(arg_i8x16, 7);
	vqshlq_n_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_s8(arg_i8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlb_n_s8(arg_i8, 0);
	vqshlb_n_s8(arg_i8, 7);
	vqshlb_n_s8(arg_i8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlb_n_s8(arg_i8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlu_n_s8(arg_i8x8, 0);
	vqshlu_n_s8(arg_i8x8, 7);
	vqshlu_n_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlu_n_s8(arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshluq_n_s8(arg_i8x16, 0);
	vqshluq_n_s8(arg_i8x16, 7);
	vqshluq_n_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshluq_n_s8(arg_i8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlub_n_s8(arg_i8, 0);
	vqshlub_n_s8(arg_i8, 7);
	vqshlub_n_s8(arg_i8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlub_n_s8(arg_i8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_s16(int16x4_t arg_i16x4, int16_t arg_i16, int16x8_t arg_i16x8) {
	vqshl_n_s16(arg_i16x4, 0);
	vqshl_n_s16(arg_i16x4, 15);
	vqshl_n_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_s16(arg_i16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_s16(arg_i16x8, 0);
	vqshlq_n_s16(arg_i16x8, 15);
	vqshlq_n_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_s16(arg_i16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlh_n_s16(arg_i16, 0);
	vqshlh_n_s16(arg_i16, 15);
	vqshlh_n_s16(arg_i16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlh_n_s16(arg_i16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlu_n_s16(arg_i16x4, 0);
	vqshlu_n_s16(arg_i16x4, 15);
	vqshlu_n_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlu_n_s16(arg_i16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshluq_n_s16(arg_i16x8, 0);
	vqshluq_n_s16(arg_i16x8, 15);
	vqshluq_n_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshluq_n_s16(arg_i16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshluh_n_s16(arg_i16, 0);
	vqshluh_n_s16(arg_i16, 15);
	vqshluh_n_s16(arg_i16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshluh_n_s16(arg_i16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_s32(int32x2_t arg_i32x2, int32_t arg_i32, int32x4_t arg_i32x4) {
	vqshl_n_s32(arg_i32x2, 0);
	vqshl_n_s32(arg_i32x2, 31);
	vqshl_n_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_s32(arg_i32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_s32(arg_i32x4, 0);
	vqshlq_n_s32(arg_i32x4, 31);
	vqshlq_n_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_s32(arg_i32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshls_n_s32(arg_i32, 0);
	vqshls_n_s32(arg_i32, 31);
	vqshls_n_s32(arg_i32, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshls_n_s32(arg_i32, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlu_n_s32(arg_i32x2, 0);
	vqshlu_n_s32(arg_i32x2, 31);
	vqshlu_n_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlu_n_s32(arg_i32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshluq_n_s32(arg_i32x4, 0);
	vqshluq_n_s32(arg_i32x4, 31);
	vqshluq_n_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshluq_n_s32(arg_i32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlus_n_s32(arg_i32, 0);
	vqshlus_n_s32(arg_i32, 31);
	vqshlus_n_s32(arg_i32, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlus_n_s32(arg_i32, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_s64(int64_t arg_i64, int64x2_t arg_i64x2, int64x1_t arg_i64x1) {
	vqshl_n_s64(arg_i64x1, 0);
	vqshl_n_s64(arg_i64x1, 63);
	vqshl_n_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_s64(arg_i64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_s64(arg_i64x2, 0);
	vqshlq_n_s64(arg_i64x2, 63);
	vqshlq_n_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_s64(arg_i64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshld_n_s64(arg_i64, 0);
	vqshld_n_s64(arg_i64, 63);
	vqshld_n_s64(arg_i64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshld_n_s64(arg_i64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlu_n_s64(arg_i64x1, 0);
	vqshlu_n_s64(arg_i64x1, 63);
	vqshlu_n_s64(arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlu_n_s64(arg_i64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshluq_n_s64(arg_i64x2, 0);
	vqshluq_n_s64(arg_i64x2, 63);
	vqshluq_n_s64(arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshluq_n_s64(arg_i64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlud_n_s64(arg_i64, 0);
	vqshlud_n_s64(arg_i64, 63);
	vqshlud_n_s64(arg_i64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlud_n_s64(arg_i64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_u8(uint8x8_t arg_u8x8, uint8_t arg_u8, uint8x16_t arg_u8x16) {
	vqshl_n_u8(arg_u8x8, 0);
	vqshl_n_u8(arg_u8x8, 7);
	vqshl_n_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_u8(arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_u8(arg_u8x16, 0);
	vqshlq_n_u8(arg_u8x16, 7);
	vqshlq_n_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_u8(arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlb_n_u8(arg_u8, 0);
	vqshlb_n_u8(arg_u8, 7);
	vqshlb_n_u8(arg_u8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlb_n_u8(arg_u8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_u16(uint16_t arg_u16, uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vqshl_n_u16(arg_u16x4, 0);
	vqshl_n_u16(arg_u16x4, 15);
	vqshl_n_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_u16(arg_u16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_u16(arg_u16x8, 0);
	vqshlq_n_u16(arg_u16x8, 15);
	vqshlq_n_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_u16(arg_u16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlh_n_u16(arg_u16, 0);
	vqshlh_n_u16(arg_u16, 15);
	vqshlh_n_u16(arg_u16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlh_n_u16(arg_u16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4, uint32_t arg_u32) {
	vqshl_n_u32(arg_u32x2, 0);
	vqshl_n_u32(arg_u32x2, 31);
	vqshl_n_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_u32(arg_u32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_u32(arg_u32x4, 0);
	vqshlq_n_u32(arg_u32x4, 31);
	vqshlq_n_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_u32(arg_u32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshls_n_u32(arg_u32, 0);
	vqshls_n_u32(arg_u32, 31);
	vqshls_n_u32(arg_u32, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshls_n_u32(arg_u32, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_saturating_shift_left_u64(uint64x1_t arg_u64x1, uint64_t arg_u64, uint64x2_t arg_u64x2) {
	vqshl_n_u64(arg_u64x1, 0);
	vqshl_n_u64(arg_u64x1, 63);
	vqshl_n_u64(arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshl_n_u64(arg_u64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshlq_n_u64(arg_u64x2, 0);
	vqshlq_n_u64(arg_u64x2, 63);
	vqshlq_n_u64(arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshlq_n_u64(arg_u64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqshld_n_u64(arg_u64, 0);
	vqshld_n_u64(arg_u64, 63);
	vqshld_n_u64(arg_u64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqshld_n_u64(arg_u64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_widen_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vshll_n_s8(arg_i8x8, 0);
	vshll_n_s8(arg_i8x8, 7);
	vshll_n_s8(arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}


	vshll_high_n_s8(arg_i8x16, 0);
	vshll_high_n_s8(arg_i8x16, 7);
	vshll_high_n_s8(arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vector_shift_left_and_widen_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vshll_n_s16(arg_i16x4, 0);
	vshll_n_s16(arg_i16x4, 15);
	vshll_n_s16(arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshll_high_n_s16(arg_i16x8, 0);
	vshll_high_n_s16(arg_i16x8, 15);
	vshll_high_n_s16(arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vector_shift_left_and_widen_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vshll_n_s32(arg_i32x2, 0);
	vshll_n_s32(arg_i32x2, 31);
	vshll_n_s32(arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshll_high_n_s32(arg_i32x4, 0);
	vshll_high_n_s32(arg_i32x4, 31);
	vshll_high_n_s32(arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vector_shift_left_and_widen_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vshll_n_u8(arg_u8x8, 0);
	vshll_n_u8(arg_u8x8, 7);
	vshll_n_u8(arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshll_high_n_u8(arg_u8x16, 0);
	vshll_high_n_u8(arg_u8x16, 7);
	vshll_high_n_u8(arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vector_shift_left_and_widen_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vshll_n_u16(arg_u16x4, 0);
	vshll_n_u16(arg_u16x4, 15);
	vshll_n_u16(arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshll_high_n_u16(arg_u16x8, 0);
	vshll_high_n_u16(arg_u16x8, 15);
	vshll_high_n_u16(arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vector_shift_left_and_widen_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vshll_n_u32(arg_u32x2, 0);
	vshll_n_u32(arg_u32x2, 31);
	vshll_n_u32(arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vshll_high_n_u32(arg_u32x4, 0);
	vshll_high_n_u32(arg_u32x4, 31);
	vshll_high_n_u32(arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vector_shift_left_and_insert_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vsli_n_s8(arg_i8x8, arg_i8x8, 0);
	vsli_n_s8(arg_i8x8, arg_i8x8, 7);
	vsli_n_s8(arg_i8x8, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_s8(arg_i8x8, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_s8(arg_i8x16, arg_i8x16, 0);
	vsliq_n_s8(arg_i8x16, arg_i8x16, 7);
	vsliq_n_s8(arg_i8x16, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_s8(arg_i8x16, arg_i8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vsli_n_s16(arg_i16x4, arg_i16x4, 0);
	vsli_n_s16(arg_i16x4, arg_i16x4, 15);
	vsli_n_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_s16(arg_i16x4, arg_i16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_s16(arg_i16x8, arg_i16x8, 0);
	vsliq_n_s16(arg_i16x8, arg_i16x8, 15);
	vsliq_n_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_s16(arg_i16x8, arg_i16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vsli_n_s32(arg_i32x2, arg_i32x2, 0);
	vsli_n_s32(arg_i32x2, arg_i32x2, 31);
	vsli_n_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_s32(arg_i32x2, arg_i32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_s32(arg_i32x4, arg_i32x4, 0);
	vsliq_n_s32(arg_i32x4, arg_i32x4, 31);
	vsliq_n_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_s32(arg_i32x4, arg_i32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_s64(int64_t arg_i64, int64x2_t arg_i64x2, int64x1_t arg_i64x1) {
	vsli_n_s64(arg_i64x1, arg_i64x1, 0);
	vsli_n_s64(arg_i64x1, arg_i64x1, 63);
	vsli_n_s64(arg_i64x1, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_s64(arg_i64x1, arg_i64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_s64(arg_i64x2, arg_i64x2, 0);
	vsliq_n_s64(arg_i64x2, arg_i64x2, 63);
	vsliq_n_s64(arg_i64x2, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_s64(arg_i64x2, arg_i64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vslid_n_s64(arg_i64, arg_i64, 0);
	vslid_n_s64(arg_i64, arg_i64, 63);
	vslid_n_s64(arg_i64, arg_i64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vslid_n_s64(arg_i64, arg_i64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vsli_n_u8(arg_u8x8, arg_u8x8, 0);
	vsli_n_u8(arg_u8x8, arg_u8x8, 7);
	vsli_n_u8(arg_u8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_u8(arg_u8x8, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_u8(arg_u8x16, arg_u8x16, 0);
	vsliq_n_u8(arg_u8x16, arg_u8x16, 7);
	vsliq_n_u8(arg_u8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_u8(arg_u8x16, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vsli_n_u16(arg_u16x4, arg_u16x4, 0);
	vsli_n_u16(arg_u16x4, arg_u16x4, 15);
	vsli_n_u16(arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_u16(arg_u16x4, arg_u16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_u16(arg_u16x8, arg_u16x8, 0);
	vsliq_n_u16(arg_u16x8, arg_u16x8, 15);
	vsliq_n_u16(arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_u16(arg_u16x8, arg_u16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vsli_n_u32(arg_u32x2, arg_u32x2, 0);
	vsli_n_u32(arg_u32x2, arg_u32x2, 31);
	vsli_n_u32(arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_u32(arg_u32x2, arg_u32x2, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_u32(arg_u32x4, arg_u32x4, 0);
	vsliq_n_u32(arg_u32x4, arg_u32x4, 31);
	vsliq_n_u32(arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_u32(arg_u32x4, arg_u32x4, 32); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_u64(uint64x1_t arg_u64x1, uint64_t arg_u64, uint64x2_t arg_u64x2) {
	vsli_n_u64(arg_u64x1, arg_u64x1, 0);
	vsli_n_u64(arg_u64x1, arg_u64x1, 63);
	vsli_n_u64(arg_u64x1, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_u64(arg_u64x1, arg_u64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_u64(arg_u64x2, arg_u64x2, 0);
	vsliq_n_u64(arg_u64x2, arg_u64x2, 63);
	vsliq_n_u64(arg_u64x2, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_u64(arg_u64x2, arg_u64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vslid_n_u64(arg_u64, arg_u64, 0);
	vslid_n_u64(arg_u64, arg_u64, 63);
	vslid_n_u64(arg_u64, arg_u64, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vslid_n_u64(arg_u64, arg_u64, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_p64(poly64x2_t arg_p64x2, poly64x1_t arg_p64x1) {
	vsli_n_p64(arg_p64x1, arg_p64x1, 0);
	vsli_n_p64(arg_p64x1, arg_p64x1, 63);
	vsli_n_p64(arg_p64x1, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_p64(arg_p64x1, arg_p64x1, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_p64(arg_p64x2, arg_p64x2, 0);
	vsliq_n_p64(arg_p64x2, arg_p64x2, 63);
	vsliq_n_p64(arg_p64x2, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_p64(arg_p64x2, arg_p64x2, 64); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_p8(poly8x16_t arg_p8x16, poly8x8_t arg_p8x8) {
	vsli_n_p8(arg_p8x8, arg_p8x8, 0);
	vsli_n_p8(arg_p8x8, arg_p8x8, 7);
	vsli_n_p8(arg_p8x8, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_p8(arg_p8x8, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_p8(arg_p8x16, arg_p8x16, 0);
	vsliq_n_p8(arg_p8x16, arg_p8x16, 7);
	vsliq_n_p8(arg_p8x16, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_p8(arg_p8x16, arg_p8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_shift_left_and_insert_p16(poly16x4_t arg_p16x4, poly16x8_t arg_p16x8) {
	vsli_n_p16(arg_p16x4, arg_p16x4, 0);
	vsli_n_p16(arg_p16x4, arg_p16x4, 15);
	vsli_n_p16(arg_p16x4, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsli_n_p16(arg_p16x4, arg_p16x4, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsliq_n_p16(arg_p16x8, arg_p16x8, 0);
	vsliq_n_p16(arg_p16x8, arg_p16x8, 15);
	vsliq_n_p16(arg_p16x8, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsliq_n_p16(arg_p16x8, arg_p16x8, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

