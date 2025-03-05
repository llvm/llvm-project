// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_vector_multiply_by_scalar_and_widen_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8) {
	vmull_lane_s16(arg_i16x4, arg_i16x4, 0);
	vmull_lane_s16(arg_i16x4, arg_i16x4, 3);
	vmull_lane_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_lane_s16(arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_lane_s16(arg_i16x8, arg_i16x4, 0);
	vmull_high_lane_s16(arg_i16x8, arg_i16x4, 3);
	vmull_high_lane_s16(arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_lane_s16(arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_laneq_s16(arg_i16x4, arg_i16x8, 0);
	vmull_laneq_s16(arg_i16x4, arg_i16x8, 7);
	vmull_laneq_s16(arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_laneq_s16(arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_laneq_s16(arg_i16x8, arg_i16x8, 0);
	vmull_high_laneq_s16(arg_i16x8, arg_i16x8, 7);
	vmull_high_laneq_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_laneq_s16(arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_and_widen_s32(int32x4_t arg_i32x4, int32x2_t arg_i32x2) {
	vmull_lane_s32(arg_i32x2, arg_i32x2, 0);
	vmull_lane_s32(arg_i32x2, arg_i32x2, 1);
	vmull_lane_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_lane_s32(arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_lane_s32(arg_i32x4, arg_i32x2, 0);
	vmull_high_lane_s32(arg_i32x4, arg_i32x2, 1);
	vmull_high_lane_s32(arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_lane_s32(arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_laneq_s32(arg_i32x2, arg_i32x4, 0);
	vmull_laneq_s32(arg_i32x2, arg_i32x4, 3);
	vmull_laneq_s32(arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_laneq_s32(arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_laneq_s32(arg_i32x4, arg_i32x4, 0);
	vmull_high_laneq_s32(arg_i32x4, arg_i32x4, 3);
	vmull_high_laneq_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_laneq_s32(arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_and_widen_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vmull_lane_u16(arg_u16x4, arg_u16x4, 0);
	vmull_lane_u16(arg_u16x4, arg_u16x4, 3);
	vmull_lane_u16(arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_lane_u16(arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_lane_u16(arg_u16x8, arg_u16x4, 0);
	vmull_high_lane_u16(arg_u16x8, arg_u16x4, 3);
	vmull_high_lane_u16(arg_u16x8, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_lane_u16(arg_u16x8, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_laneq_u16(arg_u16x4, arg_u16x8, 0);
	vmull_laneq_u16(arg_u16x4, arg_u16x8, 7);
	vmull_laneq_u16(arg_u16x4, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_laneq_u16(arg_u16x4, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_laneq_u16(arg_u16x8, arg_u16x8, 0);
	vmull_high_laneq_u16(arg_u16x8, arg_u16x8, 7);
	vmull_high_laneq_u16(arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_laneq_u16(arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_by_scalar_and_widen_u32(uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vmull_lane_u32(arg_u32x2, arg_u32x2, 0);
	vmull_lane_u32(arg_u32x2, arg_u32x2, 1);
	vmull_lane_u32(arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_lane_u32(arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_lane_u32(arg_u32x4, arg_u32x2, 0);
	vmull_high_lane_u32(arg_u32x4, arg_u32x2, 1);
	vmull_high_lane_u32(arg_u32x4, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_lane_u32(arg_u32x4, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_laneq_u32(arg_u32x2, arg_u32x4, 0);
	vmull_laneq_u32(arg_u32x2, arg_u32x4, 3);
	vmull_laneq_u32(arg_u32x2, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_laneq_u32(arg_u32x2, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmull_high_laneq_u32(arg_u32x4, arg_u32x4, 0);
	vmull_high_laneq_u32(arg_u32x4, arg_u32x4, 3);
	vmull_high_laneq_u32(arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmull_high_laneq_u32(arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

