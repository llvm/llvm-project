// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

void test_extract_vector_from_a_pair_of_vectors_s8(int8x8_t arg_i8x8, int8x16_t arg_i8x16) {
	vext_s8(arg_i8x8, arg_i8x8, 0);
	vext_s8(arg_i8x8, arg_i8x8, 7);
	vext_s8(arg_i8x8, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_s8(arg_i8x8, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_s8(arg_i8x16, arg_i8x16, 0);
	vextq_s8(arg_i8x16, arg_i8x16, 15);
	vextq_s8(arg_i8x16, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_s8(arg_i8x16, arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_s16(int16x8_t arg_i16x8, int16x4_t arg_i16x4) {
	vext_s16(arg_i16x4, arg_i16x4, 0);
	vext_s16(arg_i16x4, arg_i16x4, 3);
	vext_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_s16(arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_s16(arg_i16x8, arg_i16x8, 0);
	vextq_s16(arg_i16x8, arg_i16x8, 7);
	vextq_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_s16(arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vext_s32(arg_i32x2, arg_i32x2, 0);
	vext_s32(arg_i32x2, arg_i32x2, 1);
	vext_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_s32(arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_s32(arg_i32x4, arg_i32x4, 0);
	vextq_s32(arg_i32x4, arg_i32x4, 3);
	vextq_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_s32(arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_s64(int64x2_t arg_i64x2, int64x1_t arg_i64x1) {
	vext_s64(arg_i64x1, arg_i64x1, 0);
	vext_s64(arg_i64x1, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_s64(arg_i64x1, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_s64(arg_i64x2, arg_i64x2, 0);
	vextq_s64(arg_i64x2, arg_i64x2, 1);
	vextq_s64(arg_i64x2, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_s64(arg_i64x2, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vext_u8(arg_u8x8, arg_u8x8, 0);
	vext_u8(arg_u8x8, arg_u8x8, 7);
	vext_u8(arg_u8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_u8(arg_u8x8, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_u8(arg_u8x16, arg_u8x16, 0);
	vextq_u8(arg_u8x16, arg_u8x16, 15);
	vextq_u8(arg_u8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_u8(arg_u8x16, arg_u8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8) {
	vext_u16(arg_u16x4, arg_u16x4, 0);
	vext_u16(arg_u16x4, arg_u16x4, 3);
	vext_u16(arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_u16(arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_u16(arg_u16x8, arg_u16x8, 0);
	vextq_u16(arg_u16x8, arg_u16x8, 7);
	vextq_u16(arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_u16(arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vext_u32(arg_u32x2, arg_u32x2, 0);
	vext_u32(arg_u32x2, arg_u32x2, 1);
	vext_u32(arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_u32(arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_u32(arg_u32x4, arg_u32x4, 0);
	vextq_u32(arg_u32x4, arg_u32x4, 3);
	vextq_u32(arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_u32(arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_u64(uint64x1_t arg_u64x1, uint64x2_t arg_u64x2) {
	vext_u64(arg_u64x1, arg_u64x1, 0);
	vext_u64(arg_u64x1, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_u64(arg_u64x1, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_u64(arg_u64x2, arg_u64x2, 0);
	vextq_u64(arg_u64x2, arg_u64x2, 1);
	vextq_u64(arg_u64x2, arg_u64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_u64(arg_u64x2, arg_u64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_p64(poly64x2_t arg_p64x2, poly64x1_t arg_p64x1) {
	vext_p64(arg_p64x1, arg_p64x1, 0);
	vext_p64(arg_p64x1, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_p64(arg_p64x1, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_p64(arg_p64x2, arg_p64x2, 0);
	vextq_p64(arg_p64x2, arg_p64x2, 1);
	vextq_p64(arg_p64x2, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_p64(arg_p64x2, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_f32(float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vext_f32(arg_f32x2, arg_f32x2, 0);
	vext_f32(arg_f32x2, arg_f32x2, 1);
	vext_f32(arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_f32(arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_f32(arg_f32x4, arg_f32x4, 0);
	vextq_f32(arg_f32x4, arg_f32x4, 3);
	vextq_f32(arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_f32(arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_f64(float64x2_t arg_f64x2, float64x1_t arg_f64x1) {
	vext_f64(arg_f64x1, arg_f64x1, 0);
	vext_f64(arg_f64x1, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_f64(arg_f64x1, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_f64(arg_f64x2, arg_f64x2, 0);
	vextq_f64(arg_f64x2, arg_f64x2, 1);
	vextq_f64(arg_f64x2, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_f64(arg_f64x2, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_p8(poly8x8_t arg_p8x8, poly8x16_t arg_p8x16) {
	vext_p8(arg_p8x8, arg_p8x8, 0);
	vext_p8(arg_p8x8, arg_p8x8, 7);
	vext_p8(arg_p8x8, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_p8(arg_p8x8, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_p8(arg_p8x16, arg_p8x16, 0);
	vextq_p8(arg_p8x16, arg_p8x16, 15);
	vextq_p8(arg_p8x16, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_p8(arg_p8x16, arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_extract_vector_from_a_pair_of_vectors_p16(poly16x8_t arg_p16x8, poly16x4_t arg_p16x4) {
	vext_p16(arg_p16x4, arg_p16x4, 0);
	vext_p16(arg_p16x4, arg_p16x4, 3);
	vext_p16(arg_p16x4, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vext_p16(arg_p16x4, arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vextq_p16(arg_p16x8, arg_p16x8, 0);
	vextq_p16(arg_p16x8, arg_p16x8, 7);
	vextq_p16(arg_p16x8, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vextq_p16(arg_p16x8, arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

