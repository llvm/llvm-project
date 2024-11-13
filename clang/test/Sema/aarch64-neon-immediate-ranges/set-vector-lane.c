// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

// vsetq_lane_u8, vsetq_lane_u16, vsetq_lane_u32, vsetq_lane_u64 are
// tesed under clang/test/Sema/arm-mve-immediates.c

void test_set_vector_lane_u8(uint8x16_t arg_u8x16, uint8_t arg_u8, uint8x8_t arg_u8x8) {
	vset_lane_u8(arg_u8, arg_u8x8, 0);
	vset_lane_u8(arg_u8, arg_u8x8, 7);
	vset_lane_u8(arg_u8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_u8(arg_u8, arg_u8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_set_vector_lane_u16(uint16x4_t arg_u16x4, uint16_t arg_u16, uint16x8_t arg_u16x8) {
	vset_lane_u16(arg_u16, arg_u16x4, 0);
	vset_lane_u16(arg_u16, arg_u16x4, 3);
	vset_lane_u16(arg_u16, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_u16(arg_u16, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_set_vector_lane_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4, uint32_t arg_u32) {
	vset_lane_u32(arg_u32, arg_u32x2, 0);
	vset_lane_u32(arg_u32, arg_u32x2, 1);
	vset_lane_u32(arg_u32, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_u32(arg_u32, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_set_vector_lane_u64(uint64x2_t arg_u64x2, uint64x1_t arg_u64x1, uint64_t arg_u64) {
	vset_lane_u64(arg_u64, arg_u64x1, 0);
	vset_lane_u64(arg_u64, arg_u64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_u64(arg_u64, arg_u64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_set_vector_lane_p64(poly64_t arg_p64, poly64x1_t arg_p64x1, poly64x2_t arg_p64x2) {
	vset_lane_p64(arg_p64, arg_p64x1, 0);
	vset_lane_p64(arg_p64, arg_p64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_p64(arg_p64, arg_p64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_p64(arg_p64, arg_p64x2, 0);
	vsetq_lane_p64(arg_p64, arg_p64x2, 1);
	vsetq_lane_p64(arg_p64, arg_p64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_p64(arg_p64, arg_p64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_s8(int8x16_t arg_i8x16, int8x8_t arg_i8x8, int8_t arg_i8) {
	vset_lane_s8(arg_i8, arg_i8x8, 0);
	vset_lane_s8(arg_i8, arg_i8x8, 7);
	vset_lane_s8(arg_i8, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_s8(arg_i8, arg_i8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_s8(arg_i8, arg_i8x16, 0);
	vsetq_lane_s8(arg_i8, arg_i8x16, 15);
	vsetq_lane_s8(arg_i8, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_s8(arg_i8, arg_i8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_s16(int16x4_t arg_i16x4, int16_t arg_i16, int16x8_t arg_i16x8) {
	vset_lane_s16(arg_i16, arg_i16x4, 0);
	vset_lane_s16(arg_i16, arg_i16x4, 3);
	vset_lane_s16(arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_s16(arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_s16(arg_i16, arg_i16x8, 0);
	vsetq_lane_s16(arg_i16, arg_i16x8, 7);
	vsetq_lane_s16(arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_s16(arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_s32(int32_t arg_i32, int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vset_lane_s32(arg_i32, arg_i32x2, 0);
	vset_lane_s32(arg_i32, arg_i32x2, 1);
	vset_lane_s32(arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_s32(arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_s32(arg_i32, arg_i32x4, 0);
	vsetq_lane_s32(arg_i32, arg_i32x4, 3);
	vsetq_lane_s32(arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_s32(arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_s64(int64_t arg_i64, int64x2_t arg_i64x2, int64x1_t arg_i64x1) {
	vset_lane_s64(arg_i64, arg_i64x1, 0);
	vset_lane_s64(arg_i64, arg_i64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_s64(arg_i64, arg_i64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_s64(arg_i64, arg_i64x2, 0);
	vsetq_lane_s64(arg_i64, arg_i64x2, 1);
	vsetq_lane_s64(arg_i64, arg_i64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_s64(arg_i64, arg_i64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_p8(poly8_t arg_p8, poly8x16_t arg_p8x16, poly8x8_t arg_p8x8) {
	vset_lane_p8(arg_p8, arg_p8x8, 0);
	vset_lane_p8(arg_p8, arg_p8x8, 7);
	vset_lane_p8(arg_p8, arg_p8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_p8(arg_p8, arg_p8x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_p8(arg_p8, arg_p8x16, 0);
	vsetq_lane_p8(arg_p8, arg_p8x16, 15);
	vsetq_lane_p8(arg_p8, arg_p8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_p8(arg_p8, arg_p8x16, 16); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_p16(poly16x4_t arg_p16x4, poly16_t arg_p16, poly16x8_t arg_p16x8) {
	vset_lane_p16(arg_p16, arg_p16x4, 0);
	vset_lane_p16(arg_p16, arg_p16x4, 3);
	vset_lane_p16(arg_p16, arg_p16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_p16(arg_p16, arg_p16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_p16(arg_p16, arg_p16x8, 0);
	vsetq_lane_p16(arg_p16, arg_p16x8, 7);
	vsetq_lane_p16(arg_p16, arg_p16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_p16(arg_p16, arg_p16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_f16(float16x8_t arg_f16x8, float16x4_t arg_f16x4, float16_t arg_f16) {
	vset_lane_f16(arg_f16, arg_f16x4, 0);
	vset_lane_f16(arg_f16, arg_f16x4, 3);
	vset_lane_f16(arg_f16, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_f16(arg_f16, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_f16(arg_f16, arg_f16x8, 0);
	vsetq_lane_f16(arg_f16, arg_f16x8, 7);
	vsetq_lane_f16(arg_f16, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_f16(arg_f16, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_f32(float32x2_t arg_f32x2, float32x4_t arg_f32x4, float32_t arg_f32) {
	vset_lane_f32(arg_f32, arg_f32x2, 0);
	vset_lane_f32(arg_f32, arg_f32x2, 1);
	vset_lane_f32(arg_f32, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_f32(arg_f32, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_f32(arg_f32, arg_f32x4, 0);
	vsetq_lane_f32(arg_f32, arg_f32x4, 3);
	vsetq_lane_f32(arg_f32, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_f32(arg_f32, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_f64(float64x1_t arg_f64x1, float64x2_t arg_f64x2, float64_t arg_f64) {
	vset_lane_f64(arg_f64, arg_f64x1, 0);
	vset_lane_f64(arg_f64, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_f64(arg_f64, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_f64(arg_f64, arg_f64x2, 0);
	vsetq_lane_f64(arg_f64, arg_f64x2, 1);
	vsetq_lane_f64(arg_f64, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_f64(arg_f64, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

