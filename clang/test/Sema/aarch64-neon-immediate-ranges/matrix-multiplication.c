// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.6a -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_dot_product_s32(int8x8_t arg_i8x8, int32x2_t arg_i32x2, uint8x16_t arg_u8x16, uint8x8_t arg_u8x8,
						  int32x4_t arg_i32x4, int8x16_t arg_i8x16) {
	vusdot_lane_s32(arg_i32x2, arg_u8x8, arg_i8x8, 0);
	vusdot_lane_s32(arg_i32x2, arg_u8x8, arg_i8x8, 1);
	vusdot_lane_s32(arg_i32x2, arg_u8x8, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vusdot_lane_s32(arg_i32x2, arg_u8x8, arg_i8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsudot_lane_s32(arg_i32x2, arg_i8x8, arg_u8x8, 0);
	vsudot_lane_s32(arg_i32x2, arg_i8x8, arg_u8x8, 1);
	vsudot_lane_s32(arg_i32x2, arg_i8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsudot_lane_s32(arg_i32x2, arg_i8x8, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vusdot_laneq_s32(arg_i32x2, arg_u8x8, arg_i8x16, 0);
	vusdot_laneq_s32(arg_i32x2, arg_u8x8, arg_i8x16, 3);
	vusdot_laneq_s32(arg_i32x2, arg_u8x8, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vusdot_laneq_s32(arg_i32x2, arg_u8x8, arg_i8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsudot_laneq_s32(arg_i32x2, arg_i8x8, arg_u8x16, 0);
	vsudot_laneq_s32(arg_i32x2, arg_i8x8, arg_u8x16, 3);
	vsudot_laneq_s32(arg_i32x2, arg_i8x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsudot_laneq_s32(arg_i32x2, arg_i8x8, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vusdotq_lane_s32(arg_i32x4, arg_u8x16, arg_i8x8, 0);
	vusdotq_lane_s32(arg_i32x4, arg_u8x16, arg_i8x8, 1);
	vusdotq_lane_s32(arg_i32x4, arg_u8x16, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vusdotq_lane_s32(arg_i32x4, arg_u8x16, arg_i8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsudotq_lane_s32(arg_i32x4, arg_i8x16, arg_u8x8, 0);
	vsudotq_lane_s32(arg_i32x4, arg_i8x16, arg_u8x8, 1);
	vsudotq_lane_s32(arg_i32x4, arg_i8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsudotq_lane_s32(arg_i32x4, arg_i8x16, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vusdotq_laneq_s32(arg_i32x4, arg_u8x16, arg_i8x16, 0);
	vusdotq_laneq_s32(arg_i32x4, arg_u8x16, arg_i8x16, 3);
	vusdotq_laneq_s32(arg_i32x4, arg_u8x16, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vusdotq_laneq_s32(arg_i32x4, arg_u8x16, arg_i8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsudotq_laneq_s32(arg_i32x4, arg_i8x16, arg_u8x16, 0);
	vsudotq_laneq_s32(arg_i32x4, arg_i8x16, arg_u8x16, 3);
	vsudotq_laneq_s32(arg_i32x4, arg_i8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsudotq_laneq_s32(arg_i32x4, arg_i8x16, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

