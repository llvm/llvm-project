// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_vector_multiply_subtract_by_scalar_s16(int16x8_t arg_i16x8, int16x4_t arg_i16x4, int32x4_t arg_i32x4) {
	vmls_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 0);
	vmls_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 3);
	vmls_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 0);
	vmlsq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 3);
	vmlsq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmls_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 0);
	vmls_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 7);
	vmls_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 0);
	vmlsq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 7);
	vmlsq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 0);
	vmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 3);
	vmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 0);
	vmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 3);
	vmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 0);
	vmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 7);
	vmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 0);
	vmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 7);
	vmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_subtract_by_scalar_s32(int64x2_t arg_i64x2, int32x4_t arg_i32x4, int32x2_t arg_i32x2) {
	vmls_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 0);
	vmls_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 1);
	vmls_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 0);
	vmlsq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 1);
	vmlsq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmls_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 0);
	vmls_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 3);
	vmls_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 0);
	vmlsq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 3);
	vmlsq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 0);
	vmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 1);
	vmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 0);
	vmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 1);
	vmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 0);
	vmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 3);
	vmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 0);
	vmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 3);
	vmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_subtract_by_scalar_u16(uint16x8_t arg_u16x8, uint16x4_t arg_u16x4, uint32x4_t arg_u32x4) {
	vmls_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, 0);
	vmls_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, 3);
	vmls_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, 0);
	vmlsq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, 3);
	vmlsq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmls_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, 0);
	vmls_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, 7);
	vmls_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, 0);
	vmlsq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, 7);
	vmlsq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, 0);
	vmlsl_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, 3);
	vmlsl_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, 0);
	vmlsl_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, 3);
	vmlsl_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, 0);
	vmlsl_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, 7);
	vmlsl_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, 0);
	vmlsl_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, 7);
	vmlsl_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_subtract_by_scalar_u32(uint64x2_t arg_u64x2, uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vmls_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, 0);
	vmls_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, 1);
	vmls_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, 0);
	vmlsq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, 1);
	vmlsq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmls_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, 0);
	vmls_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, 3);
	vmls_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, 0);
	vmlsq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, 3);
	vmlsq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, 0);
	vmlsl_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, 1);
	vmlsl_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, 0);
	vmlsl_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, 1);
	vmlsl_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, 0);
	vmlsl_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, 3);
	vmlsl_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsl_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, 0);
	vmlsl_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, 3);
	vmlsl_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsl_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_subtract_by_scalar_f32(float32x4_t arg_f32x4, float32x2_t arg_f32x2) {
	vmls_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 0);
	vmls_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 1);
	vmls_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 0);
	vmlsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 1);
	vmlsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmls_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 0);
	vmls_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 3);
	vmls_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmls_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 0);
	vmlsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 3);
	vmlsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlsq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

