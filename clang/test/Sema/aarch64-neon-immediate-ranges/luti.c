// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +lut -target-feature +bf16 -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

// 2-bit indices

void test_lookup_read_2bit_u8(uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vluti2_lane_u8(arg_u8x8, arg_u8x8, 0);
	vluti2_lane_u8(arg_u8x8, arg_u8x8, 1);
	vluti2_lane_u8(arg_u8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_u8(arg_u8x8, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_u8(arg_u8x8, arg_u8x16, 0);
	vluti2_laneq_u8(arg_u8x8, arg_u8x16, 3);
	vluti2_laneq_u8(arg_u8x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_u8(arg_u8x8, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_u8(arg_u8x16, arg_u8x8, 0);
	vluti2q_lane_u8(arg_u8x16, arg_u8x8, 1);
	vluti2q_lane_u8(arg_u8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_u8(arg_u8x16, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_u8(arg_u8x16, arg_u8x16, 0);
	vluti2q_laneq_u8(arg_u8x16, arg_u8x16, 3);
	vluti2q_laneq_u8(arg_u8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_u8(arg_u8x16, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_s8(int8x8_t arg_i8x8, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8, int8x16_t arg_i8x16) {
	vluti2_lane_s8(arg_i8x8, arg_u8x8, 0);
	vluti2_lane_s8(arg_i8x8, arg_u8x8, 1);
	vluti2_lane_s8(arg_i8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_s8(arg_i8x8, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_s8(arg_i8x8, arg_u8x16, 0);
	vluti2_laneq_s8(arg_i8x8, arg_u8x16, 3);
	vluti2_laneq_s8(arg_i8x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_s8(arg_i8x8, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_s8(arg_i8x16, arg_u8x8, 0);
	vluti2q_lane_s8(arg_i8x16, arg_u8x8, 1);
	vluti2q_lane_s8(arg_i8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_s8(arg_i8x16, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_s8(arg_i8x16, arg_u8x16, 0);
	vluti2q_laneq_s8(arg_i8x16, arg_u8x16, 3);
	vluti2q_laneq_s8(arg_i8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_s8(arg_i8x16, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_p8(poly8x16_t arg_p8x16, poly8x8_t arg_p8x8, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vluti2_lane_p8(arg_p8x8, arg_u8x8, 0);
	vluti2_lane_p8(arg_p8x8, arg_u8x8, 1);
	vluti2_lane_p8(arg_p8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_p8(arg_p8x8, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_p8(arg_p8x8, arg_u8x16, 0);
	vluti2_laneq_p8(arg_p8x8, arg_u8x16, 3);
	vluti2_laneq_p8(arg_p8x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_p8(arg_p8x8, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_p8(arg_p8x16, arg_u8x8, 0);
	vluti2q_lane_p8(arg_p8x16, arg_u8x8, 1);
	vluti2q_lane_p8(arg_p8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_p8(arg_p8x16, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_p8(arg_p8x16, arg_u8x16, 0);
	vluti2q_laneq_p8(arg_p8x16, arg_u8x16, 3);
	vluti2q_laneq_p8(arg_p8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_p8(arg_p8x16, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_u16(uint16x4_t arg_u16x4, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8, uint16x8_t arg_u16x8) {
	vluti2_lane_u16(arg_u16x4, arg_u8x8, 0);
	vluti2_lane_u16(arg_u16x4, arg_u8x8, 3);
	vluti2_lane_u16(arg_u16x4, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_u16(arg_u16x4, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_u16(arg_u16x4, arg_u8x16, 0);
	vluti2_laneq_u16(arg_u16x4, arg_u8x16, 7);
	vluti2_laneq_u16(arg_u16x4, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_u16(arg_u16x4, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_u16(arg_u16x8, arg_u8x8, 0);
	vluti2q_lane_u16(arg_u16x8, arg_u8x8, 3);
	vluti2q_lane_u16(arg_u16x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_u16(arg_u16x8, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_u16(arg_u16x8, arg_u8x16, 0);
	vluti2q_laneq_u16(arg_u16x8, arg_u8x16, 7);
	vluti2q_laneq_u16(arg_u16x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_u16(arg_u16x8, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vluti2_lane_s16(arg_i16x4, arg_u8x8, 0);
	vluti2_lane_s16(arg_i16x4, arg_u8x8, 3);
	vluti2_lane_s16(arg_i16x4, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_s16(arg_i16x4, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_s16(arg_i16x4, arg_u8x16, 0);
	vluti2_laneq_s16(arg_i16x4, arg_u8x16, 7);
	vluti2_laneq_s16(arg_i16x4, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_s16(arg_i16x4, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_s16(arg_i16x8, arg_u8x8, 0);
	vluti2q_lane_s16(arg_i16x8, arg_u8x8, 3);
	vluti2q_lane_s16(arg_i16x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_s16(arg_i16x8, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_s16(arg_i16x8, arg_u8x16, 0);
	vluti2q_laneq_s16(arg_i16x8, arg_u8x16, 7);
	vluti2q_laneq_s16(arg_i16x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_s16(arg_i16x8, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_f16(float16x8_t arg_f16x8, uint8x16_t arg_u8x16, float16x4_t arg_f16x4, uint8x8_t arg_u8x8) {
	vluti2_lane_f16(arg_f16x4, arg_u8x8, 0);
	vluti2_lane_f16(arg_f16x4, arg_u8x8, 3);
	vluti2_lane_f16(arg_f16x4, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_f16(arg_f16x4, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_f16(arg_f16x4, arg_u8x16, 0);
	vluti2_laneq_f16(arg_f16x4, arg_u8x16, 7);
	vluti2_laneq_f16(arg_f16x4, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_f16(arg_f16x4, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_f16(arg_f16x8, arg_u8x8, 0);
	vluti2q_lane_f16(arg_f16x8, arg_u8x8, 3);
	vluti2q_lane_f16(arg_f16x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_f16(arg_f16x8, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_f16(arg_f16x8, arg_u8x16, 0);
	vluti2q_laneq_f16(arg_f16x8, arg_u8x16, 7);
	vluti2q_laneq_f16(arg_f16x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_f16(arg_f16x8, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_bf16(bfloat16x4_t arg_b16x4, bfloat16x8_t arg_b16x8, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vluti2_lane_bf16(arg_b16x4, arg_u8x8, 0);
	vluti2_lane_bf16(arg_b16x4, arg_u8x8, 3);
	vluti2_lane_bf16(arg_b16x4, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_bf16(arg_b16x4, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_bf16(arg_b16x4, arg_u8x16, 0);
	vluti2_laneq_bf16(arg_b16x4, arg_u8x16, 7);
	vluti2_laneq_bf16(arg_b16x4, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_bf16(arg_b16x4, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_bf16(arg_b16x8, arg_u8x8, 0);
	vluti2q_lane_bf16(arg_b16x8, arg_u8x8, 3);
	vluti2q_lane_bf16(arg_b16x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_bf16(arg_b16x8, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_bf16(arg_b16x8, arg_u8x16, 0);
	vluti2q_laneq_bf16(arg_b16x8, arg_u8x16, 7);
	vluti2q_laneq_bf16(arg_b16x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_bf16(arg_b16x8, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_2bit_p16(poly16x4_t arg_p16x4, poly16x8_t arg_p16x8, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8) {
	vluti2_lane_p16(arg_p16x4, arg_u8x8, 0);
	vluti2_lane_p16(arg_p16x4, arg_u8x8, 3);
	vluti2_lane_p16(arg_p16x4, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_lane_p16(arg_p16x4, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2_laneq_p16(arg_p16x4, arg_u8x16, 0);
	vluti2_laneq_p16(arg_p16x4, arg_u8x16, 7);
	vluti2_laneq_p16(arg_p16x4, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2_laneq_p16(arg_p16x4, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_lane_p16(arg_p16x8, arg_u8x8, 0);
	vluti2q_lane_p16(arg_p16x8, arg_u8x8, 3);
	vluti2q_lane_p16(arg_p16x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_lane_p16(arg_p16x8, arg_u8x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti2q_laneq_p16(arg_p16x8, arg_u8x16, 0);
	vluti2q_laneq_p16(arg_p16x8, arg_u8x16, 7);
	vluti2q_laneq_p16(arg_p16x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti2q_laneq_p16(arg_p16x8, arg_u8x16, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

// 4-bit indices 

void test_lookup_read_4bit_u8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vluti4q_lane_u8(arg_u8x16, arg_u8x8, 0);
	vluti4q_lane_u8(arg_u8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_u8(arg_u8x16, arg_u8x8, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_u8(arg_u8x16, arg_u8x16, 0);
	vluti4q_laneq_u8(arg_u8x16, arg_u8x16, 1);
	vluti4q_laneq_u8(arg_u8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_u8(arg_u8x16, arg_u8x16, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_4bit_s8(int8x16_t arg_i8x16, uint8x8_t arg_u8x8, uint8x16_t arg_u8x16) {
	vluti4q_lane_s8(arg_i8x16, arg_u8x8, 0);
	vluti4q_lane_s8(arg_i8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_s8(arg_i8x16, arg_u8x8, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_s8(arg_i8x16, arg_u8x16, 0);
	vluti4q_laneq_s8(arg_i8x16, arg_u8x16, 1);
	vluti4q_laneq_s8(arg_i8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_s8(arg_i8x16, arg_u8x16, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_4bit_p8(uint8x8_t arg_u8x8, uint8x16_t arg_u8x16, poly8x16_t arg_p8x16) {
	vluti4q_lane_p8(arg_p8x16, arg_u8x8, 0);
	vluti4q_lane_p8(arg_p8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_p8(arg_p8x16, arg_u8x8, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_p8(arg_p8x16, arg_u8x16, 0);
	vluti4q_laneq_p8(arg_p8x16, arg_u8x16, 1);
	vluti4q_laneq_p8(arg_p8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_p8(arg_p8x16, arg_u8x16, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_lookup_read_4bit_x2(int16x8x2_t arg_i16x8x2, uint8x8_t arg_u8x8, float16x8x2_t arg_f16x8x2, uint8x16_t arg_u8x16, poly16x8x2_t arg_p16x8x2, uint16x8x2_t arg_u16x8x2, bfloat16x8x2_t arg_b16x8x2) {
	vluti4q_lane_u16_x2(arg_u16x8x2, arg_u8x8, 0);
	vluti4q_lane_u16_x2(arg_u16x8x2, arg_u8x8, 1);
	vluti4q_lane_u16_x2(arg_u16x8x2, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_u16_x2(arg_u16x8x2, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_u16_x2(arg_u16x8x2, arg_u8x16, 0);
	vluti4q_laneq_u16_x2(arg_u16x8x2, arg_u8x16, 3);
	vluti4q_laneq_u16_x2(arg_u16x8x2, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_u16_x2(arg_u16x8x2, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_lane_s16_x2(arg_i16x8x2, arg_u8x8, 0);
	vluti4q_lane_s16_x2(arg_i16x8x2, arg_u8x8, 1);
	vluti4q_lane_s16_x2(arg_i16x8x2, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_s16_x2(arg_i16x8x2, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_s16_x2(arg_i16x8x2, arg_u8x16, 0);
	vluti4q_laneq_s16_x2(arg_i16x8x2, arg_u8x16, 3);
	vluti4q_laneq_s16_x2(arg_i16x8x2, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_s16_x2(arg_i16x8x2, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_lane_f16_x2(arg_f16x8x2, arg_u8x8, 0);
	vluti4q_lane_f16_x2(arg_f16x8x2, arg_u8x8, 1);
	vluti4q_lane_f16_x2(arg_f16x8x2, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_f16_x2(arg_f16x8x2, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_f16_x2(arg_f16x8x2, arg_u8x16, 0);
	vluti4q_laneq_f16_x2(arg_f16x8x2, arg_u8x16, 3);
	vluti4q_laneq_f16_x2(arg_f16x8x2, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_f16_x2(arg_f16x8x2, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_lane_bf16_x2(arg_b16x8x2, arg_u8x8, 0);
	vluti4q_lane_bf16_x2(arg_b16x8x2, arg_u8x8, 1);
	vluti4q_lane_bf16_x2(arg_b16x8x2, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_bf16_x2(arg_b16x8x2, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_bf16_x2(arg_b16x8x2, arg_u8x16, 0);
	vluti4q_laneq_bf16_x2(arg_b16x8x2, arg_u8x16, 3);
	vluti4q_laneq_bf16_x2(arg_b16x8x2, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_bf16_x2(arg_b16x8x2, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_lane_p16_x2(arg_p16x8x2, arg_u8x8, 0);
	vluti4q_lane_p16_x2(arg_p16x8x2, arg_u8x8, 1);
	vluti4q_lane_p16_x2(arg_p16x8x2, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_lane_p16_x2(arg_p16x8x2, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vluti4q_laneq_p16_x2(arg_p16x8x2, arg_u8x16, 0);
	vluti4q_laneq_p16_x2(arg_p16x8x2, arg_u8x16, 3);
	vluti4q_laneq_p16_x2(arg_p16x8x2, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vluti4q_laneq_p16_x2(arg_p16x8x2, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}


