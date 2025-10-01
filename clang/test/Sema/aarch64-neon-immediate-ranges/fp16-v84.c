// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.4a -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_fused_multiply_accumulate_f16(float32x2_t arg_f32x2, float32x4_t arg_f32x4, float16x4_t arg_f16x4, float16x8_t arg_f16x8) {
	vfmlal_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, 0);
	vfmlal_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, 3);
	vfmlal_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlal_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlal_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, 0);
	vfmlal_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, 7);
	vfmlal_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlal_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlalq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, 0);
	vfmlalq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, 3);
	vfmlalq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlalq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlalq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, 0);
	vfmlalq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, 7);
	vfmlalq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlalq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlsl_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, 0);
	vfmlsl_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, 3);
	vfmlsl_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlsl_lane_low_f16(arg_f32x2, arg_f16x4, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlsl_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, 0);
	vfmlsl_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, 7);
	vfmlsl_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlsl_laneq_low_f16(arg_f32x2, arg_f16x4, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlslq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, 0);
	vfmlslq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, 3);
	vfmlslq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlslq_lane_low_f16(arg_f32x4, arg_f16x8, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlslq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, 0);
	vfmlslq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, 7);
	vfmlslq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlslq_laneq_low_f16(arg_f32x4, arg_f16x8, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlal_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, 0);
	vfmlal_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, 3);
	vfmlal_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlal_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlsl_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, 0);
	vfmlsl_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, 3);
	vfmlsl_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlsl_lane_high_f16(arg_f32x2, arg_f16x4, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlalq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, 0);
	vfmlalq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, 3);
	vfmlalq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlalq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlslq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, 0);
	vfmlslq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, 3);
	vfmlslq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlslq_lane_high_f16(arg_f32x4, arg_f16x8, arg_f16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlal_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, 0);
	vfmlal_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, 7);
	vfmlal_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlal_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlsl_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, 0);
	vfmlsl_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, 7);
	vfmlsl_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlsl_laneq_high_f16(arg_f32x2, arg_f16x4, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlalq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, 0);
	vfmlalq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, 7);
	vfmlalq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlalq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vfmlslq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, 0);
	vfmlslq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, 7);
	vfmlslq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vfmlslq_laneq_high_f16(arg_f32x4, arg_f16x8, arg_f16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

}

