// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target

void test_vector_multiply_accumulate_by_scalar_s16(int32x4_t arg_i32x4, int16x8_t arg_i16x8, int16x4_t arg_i16x4) {
	vmla_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 0);
	vmla_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 3);
	vmla_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_lane_s16(arg_i16x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 0);
	vmlaq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 3);
	vmlaq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_lane_s16(arg_i16x8, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmla_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 0);
	vmla_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 7);
	vmla_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_laneq_s16(arg_i16x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 0);
	vmlaq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 7);
	vmlaq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_laneq_s16(arg_i16x8, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 0);
	vmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 3);
	vmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_lane_s16(arg_i32x4, arg_i16x4, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 0);
	vmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 3);
	vmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_lane_s16(arg_i32x4, arg_i16x8, arg_i16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 0);
	vmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 7);
	vmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_laneq_s16(arg_i32x4, arg_i16x4, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 0);
	vmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 7);
	vmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_laneq_s16(arg_i32x4, arg_i16x8, arg_i16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_accumulate_by_scalar_s32(int32x4_t arg_i32x4, int32x2_t arg_i32x2, int64x2_t arg_i64x2) {
	vmla_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 0);
	vmla_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 1);
	vmla_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_lane_s32(arg_i32x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 0);
	vmlaq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 1);
	vmlaq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_lane_s32(arg_i32x4, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmla_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 0);
	vmla_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 3);
	vmla_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_laneq_s32(arg_i32x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 0);
	vmlaq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 3);
	vmlaq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_laneq_s32(arg_i32x4, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 0);
	vmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 1);
	vmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_lane_s32(arg_i64x2, arg_i32x2, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 0);
	vmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 1);
	vmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_lane_s32(arg_i64x2, arg_i32x4, arg_i32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 0);
	vmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 3);
	vmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_laneq_s32(arg_i64x2, arg_i32x2, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 0);
	vmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 3);
	vmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_laneq_s32(arg_i64x2, arg_i32x4, arg_i32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_accumulate_by_scalar_u16(uint16x4_t arg_u16x4, uint16x8_t arg_u16x8, uint32x4_t arg_u32x4) {
	vmla_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, 0);
	vmla_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, 3);
	vmla_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_lane_u16(arg_u16x4, arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, 0);
	vmlaq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, 3);
	vmlaq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_lane_u16(arg_u16x8, arg_u16x8, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmla_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, 0);
	vmla_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, 7);
	vmla_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_laneq_u16(arg_u16x4, arg_u16x4, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, 0);
	vmlaq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, 7);
	vmlaq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_laneq_u16(arg_u16x8, arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, 0);
	vmlal_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, 3);
	vmlal_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_lane_u16(arg_u32x4, arg_u16x4, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, 0);
	vmlal_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, 3);
	vmlal_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_lane_u16(arg_u32x4, arg_u16x8, arg_u16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, 0);
	vmlal_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, 7);
	vmlal_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_laneq_u16(arg_u32x4, arg_u16x4, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, 0);
	vmlal_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, 7);
	vmlal_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_laneq_u16(arg_u32x4, arg_u16x8, arg_u16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_accumulate_by_scalar_u32(uint64x2_t arg_u64x2, uint32x2_t arg_u32x2, uint32x4_t arg_u32x4) {
	vmla_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, 0);
	vmla_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, 1);
	vmla_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_lane_u32(arg_u32x2, arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, 0);
	vmlaq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, 1);
	vmlaq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_lane_u32(arg_u32x4, arg_u32x4, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmla_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, 0);
	vmla_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, 3);
	vmla_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_laneq_u32(arg_u32x2, arg_u32x2, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, 0);
	vmlaq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, 3);
	vmlaq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_laneq_u32(arg_u32x4, arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, 0);
	vmlal_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, 1);
	vmlal_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_lane_u32(arg_u64x2, arg_u32x2, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, 0);
	vmlal_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, 1);
	vmlal_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_lane_u32(arg_u64x2, arg_u32x4, arg_u32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, 0);
	vmlal_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, 3);
	vmlal_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_laneq_u32(arg_u64x2, arg_u32x2, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlal_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, 0);
	vmlal_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, 3);
	vmlal_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlal_high_laneq_u32(arg_u64x2, arg_u32x4, arg_u32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_accumulate_by_scalar_f32(float32x4_t arg_f32x4, float32x2_t arg_f32x2) {
	vmla_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 0);
	vmla_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 1);
	vmla_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_lane_f32(arg_f32x2, arg_f32x2, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 0);
	vmlaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 1);
	vmlaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_lane_f32(arg_f32x4, arg_f32x4, arg_f32x2, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmla_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 0);
	vmla_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 3);
	vmla_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmla_laneq_f32(arg_f32x2, arg_f32x2, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vmlaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 0);
	vmlaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 3);
	vmlaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vmlaq_laneq_f32(arg_f32x4, arg_f32x4, arg_f32x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

