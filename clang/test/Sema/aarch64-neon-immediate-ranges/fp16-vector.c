// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -target-feature +v8.2a -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
#include <arm_fp16.h>
// REQUIRES: aarch64-registered-target

// vcvtq_n_f16_u16 is tested under clang/test/Sema/arm-mve-immediates.c

void test_multiplication_f16(float16_t arg_f16, float16x8_t arg_f16x8, float16x4_t arg_f16x4) {
	vmul_lane_f16(arg_f16x4, arg_f16x4, 0);
	vmul_lane_f16(arg_f16x4, arg_f16x4, 3);
	vmul_lane_f16(arg_f16x4, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_lane_f16(arg_f16x4, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_lane_f16(arg_f16x8, arg_f16x4, 0);
	vmulq_lane_f16(arg_f16x8, arg_f16x4, 3);
	vmulq_lane_f16(arg_f16x8, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_lane_f16(arg_f16x8, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmul_laneq_f16(arg_f16x4, arg_f16x8, 0);
	vmul_laneq_f16(arg_f16x4, arg_f16x8, 7);
	vmul_laneq_f16(arg_f16x4, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmul_laneq_f16(arg_f16x4, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulq_laneq_f16(arg_f16x8, arg_f16x8, 0);
	vmulq_laneq_f16(arg_f16x8, arg_f16x8, 7);
	vmulq_laneq_f16(arg_f16x8, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulq_laneq_f16(arg_f16x8, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulh_lane_f16(arg_f16, arg_f16x4, 0);
	vmulh_lane_f16(arg_f16, arg_f16x4, 3);
	vmulh_lane_f16(arg_f16, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulh_lane_f16(arg_f16, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulh_laneq_f16(arg_f16, arg_f16x8, 0);
	vmulh_laneq_f16(arg_f16, arg_f16x8, 7);
	vmulh_laneq_f16(arg_f16, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulh_laneq_f16(arg_f16, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_multiply_extended_f16(float16_t arg_f16, float16x8_t arg_f16x8, float16x4_t arg_f16x4) {
	vmulx_lane_f16(arg_f16x4, arg_f16x4, 0);
	vmulx_lane_f16(arg_f16x4, arg_f16x4, 3);
	vmulx_lane_f16(arg_f16x4, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulx_lane_f16(arg_f16x4, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxq_lane_f16(arg_f16x8, arg_f16x4, 0);
	vmulxq_lane_f16(arg_f16x8, arg_f16x4, 3);
	vmulxq_lane_f16(arg_f16x8, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxq_lane_f16(arg_f16x8, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulx_laneq_f16(arg_f16x4, arg_f16x8, 0);
	vmulx_laneq_f16(arg_f16x4, arg_f16x8, 7);
	vmulx_laneq_f16(arg_f16x4, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulx_laneq_f16(arg_f16x4, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxq_laneq_f16(arg_f16x8, arg_f16x8, 0);
	vmulxq_laneq_f16(arg_f16x8, arg_f16x8, 7);
	vmulxq_laneq_f16(arg_f16x8, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxq_laneq_f16(arg_f16x8, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxh_lane_f16(arg_f16, arg_f16x4, 0);
	vmulxh_lane_f16(arg_f16, arg_f16x4, 3);
	vmulxh_lane_f16(arg_f16, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxh_lane_f16(arg_f16, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmulxh_laneq_f16(arg_f16, arg_f16x8, 0);
	vmulxh_laneq_f16(arg_f16, arg_f16x8, 7);
	vmulxh_laneq_f16(arg_f16, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmulxh_laneq_f16(arg_f16, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_fused_multiply_accumulate_f16(float16_t arg_f16, float16x8_t arg_f16x8, float16x4_t arg_f16x4) {
	vfma_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, 0);
	vfma_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, 3);
	vfma_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfma_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmaq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, 0);
	vfmaq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, 3);
	vfmaq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmaq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfma_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, 0);
	vfma_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, 7);
	vfma_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfma_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmaq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, 0);
	vfmaq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, 7);
	vfmaq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmaq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmah_lane_f16(arg_f16, arg_f16, arg_f16x4, 0);
	vfmah_lane_f16(arg_f16, arg_f16, arg_f16x4, 3);
	vfmah_lane_f16(arg_f16, arg_f16, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmah_lane_f16(arg_f16, arg_f16, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmah_laneq_f16(arg_f16, arg_f16, arg_f16x8, 0);
	vfmah_laneq_f16(arg_f16, arg_f16, arg_f16x8, 7);
	vfmah_laneq_f16(arg_f16, arg_f16, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmah_laneq_f16(arg_f16, arg_f16, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfms_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, 0);
	vfms_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, 3);
	vfms_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfms_lane_f16(arg_f16x4, arg_f16x4, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, 0);
	vfmsq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, 3);
	vfmsq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsq_lane_f16(arg_f16x8, arg_f16x8, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfms_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, 0);
	vfms_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, 7);
	vfms_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfms_laneq_f16(arg_f16x4, arg_f16x4, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, 0);
	vfmsq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, 7);
	vfmsq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsq_laneq_f16(arg_f16x8, arg_f16x8, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsh_lane_f16(arg_f16, arg_f16, arg_f16x4, 0);
	vfmsh_lane_f16(arg_f16, arg_f16, arg_f16x4, 3);
	vfmsh_lane_f16(arg_f16, arg_f16, arg_f16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsh_lane_f16(arg_f16, arg_f16, arg_f16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vfmsh_laneq_f16(arg_f16, arg_f16, arg_f16x8, 0);
	vfmsh_laneq_f16(arg_f16, arg_f16, arg_f16x8, 7);
	vfmsh_laneq_f16(arg_f16, arg_f16, arg_f16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vfmsh_laneq_f16(arg_f16, arg_f16, arg_f16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_s16(int16x8_t arg_i16x8, int16x4_t arg_i16x4) {
	vcvt_n_f16_s16(arg_i16x4, 1);
	vcvt_n_f16_s16(arg_i16x4, 16);
	vcvt_n_f16_s16(arg_i16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_f16_s16(arg_i16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_f16_s16(arg_i16x8, 1);
	vcvtq_n_f16_s16(arg_i16x8, 16);
	vcvtq_n_f16_s16(arg_i16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_f16_s16(arg_i16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4) {
	vcvt_n_f16_u16(arg_u16x4, 1);
	vcvt_n_f16_u16(arg_u16x4, 16);
	vcvt_n_f16_u16(arg_u16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_f16_u16(arg_u16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_f16(float16x8_t arg_f16x8, float16x4_t arg_f16x4) {
	vcvt_n_s16_f16(arg_f16x4, 1);
	vcvt_n_s16_f16(arg_f16x4, 16);
	vcvt_n_s16_f16(arg_f16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_s16_f16(arg_f16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_s16_f16(arg_f16x8, 1);
	vcvtq_n_s16_f16(arg_f16x8, 16);
	vcvtq_n_s16_f16(arg_f16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_s16_f16(arg_f16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvt_n_u16_f16(arg_f16x4, 1);
	vcvt_n_u16_f16(arg_f16x4, 16);
	vcvt_n_u16_f16(arg_f16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_u16_f16(arg_f16x4, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_u16_f16(arg_f16x8, 1);
	vcvtq_n_u16_f16(arg_f16x8, 16);
	vcvtq_n_u16_f16(arg_f16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_u16_f16(arg_f16x8, 17); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

