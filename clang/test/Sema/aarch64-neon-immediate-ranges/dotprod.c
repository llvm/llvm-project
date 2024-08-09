// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.2a -target-feature +dotprod -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

// s32 variant is tested under clang/test/CodeGen/arm-neon-range-checks
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

