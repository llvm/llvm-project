// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.2a -target-feature +dotprod -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

void test_dot_product_u32(uint8x8_t arg_u8x8, uint32x2_t arg_u32x2, uint8x16_t arg_u8x16, uint32x4_t arg_u32x4) {
	vdot_lane_u32(arg_u32x2, arg_u8x8, arg_u8x8, 0);
	vdot_lane_u32(arg_u32x2, arg_u8x8, arg_u8x8, 1);
	vdot_lane_u32(arg_u32x2, arg_u8x8, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdot_lane_u32(arg_u32x2, arg_u8x8, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdotq_laneq_u32(arg_u32x4, arg_u8x16, arg_u8x16, 0);
	vdotq_laneq_u32(arg_u32x4, arg_u8x16, arg_u8x16, 3);
	vdotq_laneq_u32(arg_u32x4, arg_u8x16, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdotq_laneq_u32(arg_u32x4, arg_u8x16, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdot_laneq_u32(arg_u32x2, arg_u8x8, arg_u8x16, 0);
	vdot_laneq_u32(arg_u32x2, arg_u8x8, arg_u8x16, 3);
	vdot_laneq_u32(arg_u32x2, arg_u8x8, arg_u8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdot_laneq_u32(arg_u32x2, arg_u8x8, arg_u8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdotq_lane_u32(arg_u32x4, arg_u8x16, arg_u8x8, 0);
	vdotq_lane_u32(arg_u32x4, arg_u8x16, arg_u8x8, 1);
	vdotq_lane_u32(arg_u32x4, arg_u8x16, arg_u8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdotq_lane_u32(arg_u32x4, arg_u8x16, arg_u8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_dot_product_s32(int32x2_t arg_i32x2, int8x16_t arg_i8x16, int8x8_t arg_i8x8, int32x4_t arg_i32x4) {
	vdot_lane_s32(arg_i32x2, arg_i8x8, arg_i8x8, 0);
	vdot_lane_s32(arg_i32x2, arg_i8x8, arg_i8x8, 1);
	vdot_lane_s32(arg_i32x2, arg_i8x8, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdot_lane_s32(arg_i32x2, arg_i8x8, arg_i8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdotq_laneq_s32(arg_i32x4, arg_i8x16, arg_i8x16, 0);
	vdotq_laneq_s32(arg_i32x4, arg_i8x16, arg_i8x16, 3);
	vdotq_laneq_s32(arg_i32x4, arg_i8x16, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdotq_laneq_s32(arg_i32x4, arg_i8x16, arg_i8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdot_laneq_s32(arg_i32x2, arg_i8x8, arg_i8x16, 0);
	vdot_laneq_s32(arg_i32x2, arg_i8x8, arg_i8x16, 3);
	vdot_laneq_s32(arg_i32x2, arg_i8x8, arg_i8x16, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdot_laneq_s32(arg_i32x2, arg_i8x8, arg_i8x16, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdotq_lane_s32(arg_i32x4, arg_i8x16, arg_i8x8, 0);
	vdotq_lane_s32(arg_i32x4, arg_i8x16, arg_i8x8, 1);
	vdotq_lane_s32(arg_i32x4, arg_i8x16, arg_i8x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdotq_lane_s32(arg_i32x4, arg_i8x16, arg_i8x8, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}
