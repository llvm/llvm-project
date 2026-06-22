// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.2a -target-feature +dotprod -target-feature +f16f32dot -ffreestanding -fsyntax-only -verify %s

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

void test_dot_product_f32_f16(float32x2_t r2, float32x4_t r4, float16x4_t h4, float16x8_t h8) {
  (void)vdot_lane_f32_f16(r2, h4, h4, -1);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 1]}}
  (void)vdot_lane_f32_f16(r2, h4, h4, 2);
// expected-error@-1 {{argument value 2 is outside the valid range [0, 1]}}

  (void)vdot_laneq_f32_f16(r2, h4, h8, -1);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 3]}}
  (void)vdot_laneq_f32_f16(r2, h4, h8, 4);
// expected-error@-1 {{argument value 4 is outside the valid range [0, 3]}}

  (void)vdotq_lane_f32_f16(r4, h8, h4, -1);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 1]}}
  (void)vdotq_lane_f32_f16(r4, h8, h4, 2);
// expected-error@-1 {{argument value 2 is outside the valid range [0, 1]}}

  (void)vdotq_laneq_f32_f16(r4, h8, h8, -1);
// expected-error@-1 {{argument value -1 is outside the valid range [0, 3]}}
  (void)vdotq_laneq_f32_f16(r4, h8, h8, 4);
// expected-error@-1 {{argument value 4 is outside the valid range [0, 3]}}
}
