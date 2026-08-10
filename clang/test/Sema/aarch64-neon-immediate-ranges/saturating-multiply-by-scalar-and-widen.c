// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_saturating_multiply_by_scalar_and_widen_s16(int16x4_t arg_i16x4, int16x8_t arg_i16x8, int16_t arg_i16) {
	vqdmull_lane_s16(arg_i16x4, arg_i16x4, 0);
	vqdmull_lane_s16(arg_i16x4, arg_i16x4, 3);
	vqdmull_lane_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_lane_s16(arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmullh_lane_s16(arg_i16, arg_i16x4, 0);
	vqdmullh_lane_s16(arg_i16, arg_i16x4, 3);
	vqdmullh_lane_s16(arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmullh_lane_s16(arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmull_high_lane_s16(arg_i16x8, arg_i16x4, 0);
	vqdmull_high_lane_s16(arg_i16x8, arg_i16x4, 3);
	vqdmull_high_lane_s16(arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_high_lane_s16(arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmull_laneq_s16(arg_i16x4, arg_i16x8, 0);
	vqdmull_laneq_s16(arg_i16x4, arg_i16x8, 7);
	vqdmull_laneq_s16(arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_laneq_s16(arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmullh_laneq_s16(arg_i16, arg_i16x8, 0);
	vqdmullh_laneq_s16(arg_i16, arg_i16x8, 7);
	vqdmullh_laneq_s16(arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmullh_laneq_s16(arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmull_high_laneq_s16(arg_i16x8, arg_i16x8, 0);
	vqdmull_high_laneq_s16(arg_i16x8, arg_i16x8, 7);
	vqdmull_high_laneq_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_high_laneq_s16(arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulh_lane_s16(arg_i16x4, arg_i16x4, 0);
	vqdmulh_lane_s16(arg_i16x4, arg_i16x4, 3);
	vqdmulh_lane_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulh_lane_s16(arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhq_lane_s16(arg_i16x8, arg_i16x4, 0);
	vqdmulhq_lane_s16(arg_i16x8, arg_i16x4, 3);
	vqdmulhq_lane_s16(arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhq_lane_s16(arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhh_lane_s16(arg_i16, arg_i16x4, 0);
	vqdmulhh_lane_s16(arg_i16, arg_i16x4, 3);
	vqdmulhh_lane_s16(arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhh_lane_s16(arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulh_laneq_s16(arg_i16x4, arg_i16x8, 0);
	vqdmulh_laneq_s16(arg_i16x4, arg_i16x8, 7);
	vqdmulh_laneq_s16(arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulh_laneq_s16(arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhq_laneq_s16(arg_i16x8, arg_i16x8, 0);
	vqdmulhq_laneq_s16(arg_i16x8, arg_i16x8, 7);
	vqdmulhq_laneq_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhq_laneq_s16(arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhh_laneq_s16(arg_i16, arg_i16x8, 0);
	vqdmulhh_laneq_s16(arg_i16, arg_i16x8, 7);
	vqdmulhh_laneq_s16(arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhh_laneq_s16(arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulh_lane_s16(arg_i16x4, arg_i16x4, 0);
	vqrdmulh_lane_s16(arg_i16x4, arg_i16x4, 3);
	vqrdmulh_lane_s16(arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulh_lane_s16(arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhq_lane_s16(arg_i16x8, arg_i16x4, 0);
	vqrdmulhq_lane_s16(arg_i16x8, arg_i16x4, 3);
	vqrdmulhq_lane_s16(arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhq_lane_s16(arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhh_lane_s16(arg_i16, arg_i16x4, 0);
	vqrdmulhh_lane_s16(arg_i16, arg_i16x4, 3);
	vqrdmulhh_lane_s16(arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhh_lane_s16(arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulh_laneq_s16(arg_i16x4, arg_i16x8, 0);
	vqrdmulh_laneq_s16(arg_i16x4, arg_i16x8, 7);
	vqrdmulh_laneq_s16(arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulh_laneq_s16(arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhq_laneq_s16(arg_i16x8, arg_i16x8, 0);
	vqrdmulhq_laneq_s16(arg_i16x8, arg_i16x8, 7);
	vqrdmulhq_laneq_s16(arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhq_laneq_s16(arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhh_laneq_s16(arg_i16, arg_i16x8, 0);
	vqrdmulhh_laneq_s16(arg_i16, arg_i16x8, 7);
	vqrdmulhh_laneq_s16(arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhh_laneq_s16(arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}


void test_saturating_multiply_by_scalar_and_widen_s32(int32x2_t arg_i32x2, int32_t arg_i32, int32x4_t arg_i32x4) {
	vqdmull_lane_s32(arg_i32x2, arg_i32x2, 0);
	vqdmull_lane_s32(arg_i32x2, arg_i32x2, 1);
	vqdmull_lane_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_lane_s32(arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulls_lane_s32(arg_i32, arg_i32x2, 0);
	vqdmulls_lane_s32(arg_i32, arg_i32x2, 1);
	vqdmulls_lane_s32(arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulls_lane_s32(arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmull_high_lane_s32(arg_i32x4, arg_i32x2, 0);
	vqdmull_high_lane_s32(arg_i32x4, arg_i32x2, 1);
	vqdmull_high_lane_s32(arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_high_lane_s32(arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmull_laneq_s32(arg_i32x2, arg_i32x4, 0);
	vqdmull_laneq_s32(arg_i32x2, arg_i32x4, 3);
	vqdmull_laneq_s32(arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_laneq_s32(arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulls_laneq_s32(arg_i32, arg_i32x4, 0);
	vqdmulls_laneq_s32(arg_i32, arg_i32x4, 3);
	vqdmulls_laneq_s32(arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulls_laneq_s32(arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmull_high_laneq_s32(arg_i32x4, arg_i32x4, 0);
	vqdmull_high_laneq_s32(arg_i32x4, arg_i32x4, 3);
	vqdmull_high_laneq_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmull_high_laneq_s32(arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulh_lane_s32(arg_i32x2, arg_i32x2, 0);
	vqdmulh_lane_s32(arg_i32x2, arg_i32x2, 1);
	vqdmulh_lane_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulh_lane_s32(arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhq_lane_s32(arg_i32x4, arg_i32x2, 0);
	vqdmulhq_lane_s32(arg_i32x4, arg_i32x2, 1);
	vqdmulhq_lane_s32(arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhq_lane_s32(arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhs_lane_s32(arg_i32, arg_i32x2, 0);
	vqdmulhs_lane_s32(arg_i32, arg_i32x2, 1);
	vqdmulhs_lane_s32(arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhs_lane_s32(arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulh_laneq_s32(arg_i32x2, arg_i32x4, 0);
	vqdmulh_laneq_s32(arg_i32x2, arg_i32x4, 3);
	vqdmulh_laneq_s32(arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulh_laneq_s32(arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhq_laneq_s32(arg_i32x4, arg_i32x4, 0);
	vqdmulhq_laneq_s32(arg_i32x4, arg_i32x4, 3);
	vqdmulhq_laneq_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhq_laneq_s32(arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmulhs_laneq_s32(arg_i32, arg_i32x4, 0);
	vqdmulhs_laneq_s32(arg_i32, arg_i32x4, 3);
	vqdmulhs_laneq_s32(arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmulhs_laneq_s32(arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulh_lane_s32(arg_i32x2, arg_i32x2, 0);
	vqrdmulh_lane_s32(arg_i32x2, arg_i32x2, 1);
	vqrdmulh_lane_s32(arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulh_lane_s32(arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhq_lane_s32(arg_i32x4, arg_i32x2, 0);
	vqrdmulhq_lane_s32(arg_i32x4, arg_i32x2, 1);
	vqrdmulhq_lane_s32(arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhq_lane_s32(arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhs_lane_s32(arg_i32, arg_i32x2, 0);
	vqrdmulhs_lane_s32(arg_i32, arg_i32x2, 1);
	vqrdmulhs_lane_s32(arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhs_lane_s32(arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulh_laneq_s32(arg_i32x2, arg_i32x4, 0);
	vqrdmulh_laneq_s32(arg_i32x2, arg_i32x4, 3);
	vqrdmulh_laneq_s32(arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulh_laneq_s32(arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhq_laneq_s32(arg_i32x4, arg_i32x4, 0);
	vqrdmulhq_laneq_s32(arg_i32x4, arg_i32x4, 3);
	vqrdmulhq_laneq_s32(arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhq_laneq_s32(arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmulhs_laneq_s32(arg_i32, arg_i32x4, 0);
	vqrdmulhs_laneq_s32(arg_i32, arg_i32x4, 3);
	vqrdmulhs_laneq_s32(arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmulhs_laneq_s32(arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

