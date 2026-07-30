// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -target-feature +v8.1a -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

void test_saturating_multiply_accumulate_by_element_s16(int16x8_t arg_i16x8, int16_t arg_i16, int16x4_t arg_i16x4) {
	vqrdmlah_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 0);
	vqrdmlah_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 3);
	vqrdmlah_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlah_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 0);
	vqrdmlahq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 3);
	vqrdmlahq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlah_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 0);
	vqrdmlah_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 7);
	vqrdmlah_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlah_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 0);
	vqrdmlahq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 7);
	vqrdmlahq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlsh_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 0);
	vqrdmlsh_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 3);
	vqrdmlsh_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlsh_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 0);
	vqrdmlshq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 3);
	vqrdmlshq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlsh_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 0);
	vqrdmlsh_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 7);
	vqrdmlsh_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlsh_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 0);
	vqrdmlshq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 7);
	vqrdmlshq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahh_lane_s16(arg_i16, arg_i16, arg_i16x4, 0);
	vqrdmlahh_lane_s16(arg_i16, arg_i16, arg_i16x4, 3);
	vqrdmlahh_lane_s16(arg_i16, arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahh_lane_s16(arg_i16, arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahh_laneq_s16(arg_i16, arg_i16, arg_i16x8, 0);
	vqrdmlahh_laneq_s16(arg_i16, arg_i16, arg_i16x8, 7);
	vqrdmlahh_laneq_s16(arg_i16, arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahh_laneq_s16(arg_i16, arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshh_lane_s16(arg_i16, arg_i16, arg_i16x4, 0);
	vqrdmlshh_lane_s16(arg_i16, arg_i16, arg_i16x4, 3);
	vqrdmlshh_lane_s16(arg_i16, arg_i16, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshh_lane_s16(arg_i16, arg_i16, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshh_laneq_s16(arg_i16, arg_i16, arg_i16x8, 0);
	vqrdmlshh_laneq_s16(arg_i16, arg_i16, arg_i16x8, 7);
	vqrdmlshh_laneq_s16(arg_i16, arg_i16, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshh_laneq_s16(arg_i16, arg_i16, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_saturating_multiply_accumulate_by_element_s32(int32x2_t arg_i32x2, int32x4_t arg_i32x4, int32_t arg_i32) {
	vqrdmlah_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 0);
	vqrdmlah_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 1);
	vqrdmlah_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlah_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 0);
	vqrdmlahq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 1);
	vqrdmlahq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlah_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 0);
	vqrdmlah_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 3);
	vqrdmlah_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlah_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 0);
	vqrdmlahq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 3);
	vqrdmlahq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlsh_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 0);
	vqrdmlsh_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 1);
	vqrdmlsh_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlsh_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 0);
	vqrdmlshq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 1);
	vqrdmlshq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlsh_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 0);
	vqrdmlsh_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 3);
	vqrdmlsh_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlsh_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 0);
	vqrdmlshq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 3);
	vqrdmlshq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahs_lane_s32(arg_i32, arg_i32, arg_i32x2, 0);
	vqrdmlahs_lane_s32(arg_i32, arg_i32, arg_i32x2, 1);
	vqrdmlahs_lane_s32(arg_i32, arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahs_lane_s32(arg_i32, arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlahs_laneq_s32(arg_i32, arg_i32, arg_i32x4, 0);
	vqrdmlahs_laneq_s32(arg_i32, arg_i32, arg_i32x4, 3);
	vqrdmlahs_laneq_s32(arg_i32, arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlahs_laneq_s32(arg_i32, arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshs_lane_s32(arg_i32, arg_i32, arg_i32x2, 0);
	vqrdmlshs_lane_s32(arg_i32, arg_i32, arg_i32x2, 1);
	vqrdmlshs_lane_s32(arg_i32, arg_i32, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshs_lane_s32(arg_i32, arg_i32, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vqrdmlshs_laneq_s32(arg_i32, arg_i32, arg_i32x4, 0);
	vqrdmlshs_laneq_s32(arg_i32, arg_i32, arg_i32x4, 3);
	vqrdmlshs_laneq_s32(arg_i32, arg_i32, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vqrdmlshs_laneq_s32(arg_i32, arg_i32, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}
