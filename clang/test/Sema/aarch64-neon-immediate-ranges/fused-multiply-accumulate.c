// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_fused_multiply_accumulate_f32(float32x2_t arg_f32x2, float32_t arg_f32, float32x4_t arg_f32x4) {
	vfma_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 0);
	vfma_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 1);
	vfma_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfma_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 0);
	vfmaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 1);
	vfmaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmas_lane_f32(arg_f32, arg_f32, arg_f32x2, 0);
	vfmas_lane_f32(arg_f32, arg_f32, arg_f32x2, 1);
	vfmas_lane_f32(arg_f32, arg_f32, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmas_lane_f32(arg_f32, arg_f32, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfma_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 0);
	vfma_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 3);
	vfma_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfma_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 0);
	vfmaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 3);
	vfmaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmas_laneq_f32(arg_f32, arg_f32, arg_f32x4, 0);
	vfmas_laneq_f32(arg_f32, arg_f32, arg_f32x4, 3);
	vfmas_laneq_f32(arg_f32, arg_f32, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmas_laneq_f32(arg_f32, arg_f32, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfms_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 0);
	vfms_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 1);
	vfms_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfms_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 0);
	vfmsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 1);
	vfmsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmss_lane_f32(arg_f32, arg_f32, arg_f32x2, 0);
	vfmss_lane_f32(arg_f32, arg_f32, arg_f32x2, 1);
	vfmss_lane_f32(arg_f32, arg_f32, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmss_lane_f32(arg_f32, arg_f32, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfms_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 0);
	vfms_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 3);
	vfms_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfms_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 0);
	vfmsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 3);
	vfmsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmss_laneq_f32(arg_f32, arg_f32, arg_f32x4, 0);
	vfmss_laneq_f32(arg_f32, arg_f32, arg_f32x4, 3);
	vfmss_laneq_f32(arg_f32, arg_f32, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmss_laneq_f32(arg_f32, arg_f32, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_fused_multiply_accumulate_f64(float64x2_t arg_f64x2, float64_t arg_f64, float64x1_t arg_f64x1) {
	vfma_lane_f64(arg_f64x1, arg_f64x1, arg_f64x1, 0);
	vfma_lane_f64(arg_f64x1, arg_f64x1, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfma_lane_f64(arg_f64x1, arg_f64x1, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmaq_lane_f64(arg_f64x2, arg_f64x2, arg_f64x1, 0);
	vfmaq_lane_f64(arg_f64x2, arg_f64x2, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmaq_lane_f64(arg_f64x2, arg_f64x2, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmad_lane_f64(arg_f64, arg_f64, arg_f64x1, 0);
	vfmad_lane_f64(arg_f64, arg_f64, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmad_lane_f64(arg_f64, arg_f64, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfma_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, 0);
	vfma_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, 1);
	vfma_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfma_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmaq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, 0);
	vfmaq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, 1);
	vfmaq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmaq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmad_laneq_f64(arg_f64, arg_f64, arg_f64x2, 0);
	vfmad_laneq_f64(arg_f64, arg_f64, arg_f64x2, 1);
	vfmad_laneq_f64(arg_f64, arg_f64, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmad_laneq_f64(arg_f64, arg_f64, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfms_lane_f64(arg_f64x1, arg_f64x1, arg_f64x1, 0);
	vfms_lane_f64(arg_f64x1, arg_f64x1, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfms_lane_f64(arg_f64x1, arg_f64x1, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsq_lane_f64(arg_f64x2, arg_f64x2, arg_f64x1, 0);
	vfmsq_lane_f64(arg_f64x2, arg_f64x2, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsq_lane_f64(arg_f64x2, arg_f64x2, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsd_lane_f64(arg_f64, arg_f64, arg_f64x1, 0);
	vfmsd_lane_f64(arg_f64, arg_f64, arg_f64x1, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsd_lane_f64(arg_f64, arg_f64, arg_f64x1, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfms_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, 0);
	vfms_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, 1);
	vfms_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfms_laneq_f64(arg_f64x1, arg_f64x1, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, 0);
	vfmsq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, 1);
	vfmsq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsq_laneq_f64(arg_f64x2, arg_f64x2, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsd_laneq_f64(arg_f64, arg_f64, arg_f64x2, 0);
	vfmsd_laneq_f64(arg_f64, arg_f64, arg_f64x2, 1);
	vfmsd_laneq_f64(arg_f64, arg_f64, arg_f64x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsd_laneq_f64(arg_f64, arg_f64, arg_f64x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

