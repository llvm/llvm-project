// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_multiply_extended_f32(float32_t arg_f32, float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vmulx_lane_f32(arg_f32x2, arg_f32x2, 0);
	vmulx_lane_f32(arg_f32x2, arg_f32x2, 1);
	vmulx_lane_f32(arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulx_lane_f32(arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxq_lane_f32(arg_f32x4, arg_f32x2, 0);
	vmulxq_lane_f32(arg_f32x4, arg_f32x2, 1);
	vmulxq_lane_f32(arg_f32x4, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxq_lane_f32(arg_f32x4, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxs_lane_f32(arg_f32, arg_f32x2, 0);
	vmulxs_lane_f32(arg_f32, arg_f32x2, 1);
	vmulxs_lane_f32(arg_f32, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxs_lane_f32(arg_f32, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulx_laneq_f32(arg_f32x2, arg_f32x4, 0);
	vmulx_laneq_f32(arg_f32x2, arg_f32x4, 3);
	vmulx_laneq_f32(arg_f32x2, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulx_laneq_f32(arg_f32x2, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxq_laneq_f32(arg_f32x4, arg_f32x4, 0);
	vmulxq_laneq_f32(arg_f32x4, arg_f32x4, 3);
	vmulxq_laneq_f32(arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxq_laneq_f32(arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxs_laneq_f32(arg_f32, arg_f32x4, 0);
	vmulxs_laneq_f32(arg_f32, arg_f32x4, 3);
	vmulxs_laneq_f32(arg_f32, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxs_laneq_f32(arg_f32, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_multiply_extended_f64(float64_t arg_f64, float64x2_t arg_f64x2, float64x1_t arg_f64x1) {
	vmulx_lane_f64(arg_f64x1, arg_f64x1, 0);
	vmulx_lane_f64(arg_f64x1, arg_f64x1, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vmulx_lane_f64(arg_f64x1, arg_f64x1, 1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vmulxq_lane_f64(arg_f64x2, arg_f64x1, 0);
	vmulxq_lane_f64(arg_f64x2, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxq_lane_f64(arg_f64x2, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxd_lane_f64(arg_f64, arg_f64x1, 0);
	vmulxd_lane_f64(arg_f64, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxd_lane_f64(arg_f64, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulx_laneq_f64(arg_f64x1, arg_f64x2, 0);
	vmulx_laneq_f64(arg_f64x1, arg_f64x2, 1);
	vmulx_laneq_f64(arg_f64x1, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulx_laneq_f64(arg_f64x1, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxq_laneq_f64(arg_f64x2, arg_f64x2, 0);
	vmulxq_laneq_f64(arg_f64x2, arg_f64x2, 1);
	vmulxq_laneq_f64(arg_f64x2, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxq_laneq_f64(arg_f64x2, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxd_laneq_f64(arg_f64, arg_f64x2, 0);
	vmulxd_laneq_f64(arg_f64, arg_f64x2, 1);
	vmulxd_laneq_f64(arg_f64, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxd_laneq_f64(arg_f64, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

