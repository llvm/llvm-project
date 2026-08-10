// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_saturating_multiply_accumulate_s16(int16x4_t arg_i16x4, int32_t arg_i32, int16_t arg_i16,
											 int32x4_t arg_i32x4, int16x8_t arg_i16x8) {
	vqdmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 0);
	vqdmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 3);
	vqdmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlalh_lane_s16(arg_i32, arg_i16, arg_i16x4, 0);
	vqdmlalh_lane_s16(arg_i32, arg_i16, arg_i16x4, 3);
	vqdmlalh_lane_s16(arg_i32, arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlalh_lane_s16(arg_i32, arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 0);
	vqdmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 3);
	vqdmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 0);
	vqdmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 7);
	vqdmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlalh_laneq_s16(arg_i32, arg_i16, arg_i16x8, 0);
	vqdmlalh_laneq_s16(arg_i32, arg_i16, arg_i16x8, 7);
	vqdmlalh_laneq_s16(arg_i32, arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlalh_laneq_s16(arg_i32, arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 0);
	vqdmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 7);
	vqdmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 0);
	vqdmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 3);
	vqdmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlslh_lane_s16(arg_i32, arg_i16, arg_i16x4, 0);
	vqdmlslh_lane_s16(arg_i32, arg_i16, arg_i16x4, 3);
	vqdmlslh_lane_s16(arg_i32, arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlslh_lane_s16(arg_i32, arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 0);
	vqdmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 3);
	vqdmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 0);
	vqdmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 7);
	vqdmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlslh_laneq_s16(arg_i32, arg_i16, arg_i16x8, 0);
	vqdmlslh_laneq_s16(arg_i32, arg_i16, arg_i16x8, 7);
	vqdmlslh_laneq_s16(arg_i32, arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlslh_laneq_s16(arg_i32, arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 0);
	vqdmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 7);
	vqdmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_saturating_multiply_accumulate_s32(int32_t arg_i32, int32x4_t arg_i32x4, int64_t arg_i64, int64x2_t arg_i64x2, int32x2_t arg_i32x2) {
	vqdmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 0);
	vqdmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 1);
	vqdmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlals_lane_s32(arg_i64, arg_i32, arg_i32x2, 0);
	vqdmlals_lane_s32(arg_i64, arg_i32, arg_i32x2, 1);
	vqdmlals_lane_s32(arg_i64, arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlals_lane_s32(arg_i64, arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 0);
	vqdmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 1);
	vqdmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 0);
	vqdmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 3);
	vqdmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlals_laneq_s32(arg_i64, arg_i32, arg_i32x4, 0);
	vqdmlals_laneq_s32(arg_i64, arg_i32, arg_i32x4, 3);
	vqdmlals_laneq_s32(arg_i64, arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlals_laneq_s32(arg_i64, arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 0);
	vqdmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 3);
	vqdmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 0);
	vqdmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 1);
	vqdmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsls_lane_s32(arg_i64, arg_i32, arg_i32x2, 0);
	vqdmlsls_lane_s32(arg_i64, arg_i32, arg_i32x2, 1);
	vqdmlsls_lane_s32(arg_i64, arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsls_lane_s32(arg_i64, arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 0);
	vqdmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 1);
	vqdmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 0);
	vqdmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 3);
	vqdmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsls_laneq_s32(arg_i64, arg_i32, arg_i32x4, 0);
	vqdmlsls_laneq_s32(arg_i64, arg_i32, arg_i32x4, 3);
	vqdmlsls_laneq_s32(arg_i64, arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsls_laneq_s32(arg_i64, arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqdmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 0);
	vqdmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 3);
	vqdmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqdmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}
