// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -target-feature +bf16 -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
#include <arm_bf16.h>
// REQUIRES: aarch64-registered-target


void test_set_all_lanes_to_the_same_value_bf16(bfloat16x8_t arg_b16x8, bfloat16x4_t arg_b16x4) {
	vdup_lane_bf16(arg_b16x4, 0);
	vdup_lane_bf16(arg_b16x4, 3);
	vdup_lane_bf16(arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_lane_bf16(arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_lane_bf16(arg_b16x4, 0);
	vdupq_lane_bf16(arg_b16x4, 3);
	vdupq_lane_bf16(arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_lane_bf16(arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdup_laneq_bf16(arg_b16x8, 0);
	vdup_laneq_bf16(arg_b16x8, 7);
	vdup_laneq_bf16(arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdup_laneq_bf16(arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vdupq_laneq_bf16(arg_b16x8, 0);
	vdupq_laneq_bf16(arg_b16x8, 7);
	vdupq_laneq_bf16(arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vdupq_laneq_bf16(arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vduph_lane_bf16(arg_b16x4, 0);
	vduph_lane_bf16(arg_b16x4, 3);
	vduph_lane_bf16(arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_lane_bf16(arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vduph_laneq_bf16(arg_b16x8, 0);
	vduph_laneq_bf16(arg_b16x8, 7);
	vduph_laneq_bf16(arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vduph_laneq_bf16(arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_split_vectors_bf16(bfloat16x8_t arg_b16x8, bfloat16x4_t arg_b16x4) {
	vget_lane_bf16(arg_b16x4, 0);
	vget_lane_bf16(arg_b16x4, 3);
	vget_lane_bf16(arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vget_lane_bf16(arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vgetq_lane_bf16(arg_b16x8, 0);
	vgetq_lane_bf16(arg_b16x8, 7);
	vgetq_lane_bf16(arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vgetq_lane_bf16(arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_set_vector_lane_bf16(bfloat16x8_t arg_b16x8, bfloat16x4_t arg_b16x4, bfloat16_t arg_b16) {
	vset_lane_bf16(arg_b16, arg_b16x4, 0);
	vset_lane_bf16(arg_b16, arg_b16x4, 3);
	vset_lane_bf16(arg_b16, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vset_lane_bf16(arg_b16, arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vsetq_lane_bf16(arg_b16, arg_b16x8, 0);
	vsetq_lane_bf16(arg_b16, arg_b16x8, 7);
	vsetq_lane_bf16(arg_b16, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vsetq_lane_bf16(arg_b16, arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_copy_vector_lane_bf16(bfloat16x8_t arg_b16x8, bfloat16x4_t arg_b16x4) {
	vcopy_lane_bf16(arg_b16x4, 0, arg_b16x4, 0);
	vcopy_lane_bf16(arg_b16x4, 3, arg_b16x4, 0);
	vcopy_lane_bf16(arg_b16x4, -1, arg_b16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_bf16(arg_b16x4, 4, arg_b16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_bf16(arg_b16x4, 0, arg_b16x4, 3);
	vcopy_lane_bf16(arg_b16x4, 0, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_lane_bf16(arg_b16x4, 0, arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_lane_bf16(arg_b16x8, 0, arg_b16x4, 0);
	vcopyq_lane_bf16(arg_b16x8, 7, arg_b16x4, 0);
	vcopyq_lane_bf16(arg_b16x8, -1, arg_b16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_bf16(arg_b16x8, 8, arg_b16x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_bf16(arg_b16x8, 0, arg_b16x4, 3);
	vcopyq_lane_bf16(arg_b16x8, 0, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_lane_bf16(arg_b16x8, 0, arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopy_laneq_bf16(arg_b16x4, 0, arg_b16x8, 0);
	vcopy_laneq_bf16(arg_b16x4, 3, arg_b16x8, 0);
	vcopy_laneq_bf16(arg_b16x4, -1, arg_b16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_bf16(arg_b16x4, 4, arg_b16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_bf16(arg_b16x4, 0, arg_b16x8, 7);
	vcopy_laneq_bf16(arg_b16x4, 0, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopy_laneq_bf16(arg_b16x4, 0, arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcopyq_laneq_bf16(arg_b16x8, 0, arg_b16x8, 0);
	vcopyq_laneq_bf16(arg_b16x8, 7, arg_b16x8, 0);
	vcopyq_laneq_bf16(arg_b16x8, -1, arg_b16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_bf16(arg_b16x8, 8, arg_b16x8, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_bf16(arg_b16x8, 0, arg_b16x8, 7);
	vcopyq_laneq_bf16(arg_b16x8, 0, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcopyq_laneq_bf16(arg_b16x8, 0, arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_load_bf16(bfloat16x4_t arg_b16x4, bfloat16x4x4_t arg_b16x4x4, bfloat16x8x4_t arg_b16x8x4,
					bfloat16x4x3_t arg_b16x4x3, bfloat16x8_t arg_b16x8, bfloat16_t* arg_b16_ptr,
					bfloat16x8x3_t arg_b16x8x3, bfloat16x4x2_t arg_b16x4x2, bfloat16x8x2_t arg_b16x8x2) {
	vld1_lane_bf16(arg_b16_ptr, arg_b16x4, 0);
	vld1_lane_bf16(arg_b16_ptr, arg_b16x4, 3);
	vld1_lane_bf16(arg_b16_ptr, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1_lane_bf16(arg_b16_ptr, arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld1q_lane_bf16(arg_b16_ptr, arg_b16x8, 0);
	vld1q_lane_bf16(arg_b16_ptr, arg_b16x8, 7);
	vld1q_lane_bf16(arg_b16_ptr, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld1q_lane_bf16(arg_b16_ptr, arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2_lane_bf16(arg_b16_ptr, arg_b16x4x2, 0);
	vld2_lane_bf16(arg_b16_ptr, arg_b16x4x2, 3);
	vld2_lane_bf16(arg_b16_ptr, arg_b16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2_lane_bf16(arg_b16_ptr, arg_b16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, 0);
	vld2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, 7);
	vld2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3_lane_bf16(arg_b16_ptr, arg_b16x4x3, 0);
	vld3_lane_bf16(arg_b16_ptr, arg_b16x4x3, 3);
	vld3_lane_bf16(arg_b16_ptr, arg_b16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3_lane_bf16(arg_b16_ptr, arg_b16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, 0);
	vld3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, 7);
	vld3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4_lane_bf16(arg_b16_ptr, arg_b16x4x4, 0);
	vld4_lane_bf16(arg_b16_ptr, arg_b16x4x4, 3);
	vld4_lane_bf16(arg_b16_ptr, arg_b16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4_lane_bf16(arg_b16_ptr, arg_b16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vld4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, 0);
	vld4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, 7);
	vld4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vld4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_store_bf16(bfloat16x4_t arg_b16x4, bfloat16x4x4_t arg_b16x4x4, bfloat16x8x4_t arg_b16x8x4,
					 bfloat16x4x3_t arg_b16x4x3, bfloat16x8_t arg_b16x8, bfloat16_t* arg_b16_ptr,
					 bfloat16x8x3_t arg_b16x8x3, bfloat16x4x2_t arg_b16x4x2, bfloat16x8x2_t arg_b16x8x2) {
	vst1_lane_bf16(arg_b16_ptr, arg_b16x4, 0);
	vst1_lane_bf16(arg_b16_ptr, arg_b16x4, 3);
	vst1_lane_bf16(arg_b16_ptr, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1_lane_bf16(arg_b16_ptr, arg_b16x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst1q_lane_bf16(arg_b16_ptr, arg_b16x8, 0);
	vst1q_lane_bf16(arg_b16_ptr, arg_b16x8, 7);
	vst1q_lane_bf16(arg_b16_ptr, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst1q_lane_bf16(arg_b16_ptr, arg_b16x8, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2_lane_bf16(arg_b16_ptr, arg_b16x4x2, 0);
	vst2_lane_bf16(arg_b16_ptr, arg_b16x4x2, 3);
	vst2_lane_bf16(arg_b16_ptr, arg_b16x4x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2_lane_bf16(arg_b16_ptr, arg_b16x4x2, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, 0);
	vst2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, 7);
	vst2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst2q_lane_bf16(arg_b16_ptr, arg_b16x8x2, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3_lane_bf16(arg_b16_ptr, arg_b16x4x3, 0);
	vst3_lane_bf16(arg_b16_ptr, arg_b16x4x3, 3);
	vst3_lane_bf16(arg_b16_ptr, arg_b16x4x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3_lane_bf16(arg_b16_ptr, arg_b16x4x3, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, 0);
	vst3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, 7);
	vst3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst3q_lane_bf16(arg_b16_ptr, arg_b16x8x3, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4_lane_bf16(arg_b16_ptr, arg_b16x4x4, 0);
	vst4_lane_bf16(arg_b16_ptr, arg_b16x4x4, 3);
	vst4_lane_bf16(arg_b16_ptr, arg_b16x4x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4_lane_bf16(arg_b16_ptr, arg_b16x4x4, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vst4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, 0);
	vst4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, 7);
	vst4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vst4q_lane_bf16(arg_b16_ptr, arg_b16x8x4, 8); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_dot_product_f32(bfloat16x8_t arg_b16x8, bfloat16x4_t arg_b16x4, float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vbfdot_lane_f32(arg_f32x2, arg_b16x4, arg_b16x4, 0);
	vbfdot_lane_f32(arg_f32x2, arg_b16x4, arg_b16x4, 1);
	vbfdot_lane_f32(arg_f32x2, arg_b16x4, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vbfdot_lane_f32(arg_f32x2, arg_b16x4, arg_b16x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vbfdotq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 0);
	vbfdotq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 3);
	vbfdotq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vbfdotq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vbfdot_laneq_f32(arg_f32x2, arg_b16x4, arg_b16x8, 0);
	vbfdot_laneq_f32(arg_f32x2, arg_b16x4, arg_b16x8, 3);
	vbfdot_laneq_f32(arg_f32x2, arg_b16x4, arg_b16x8, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vbfdot_laneq_f32(arg_f32x2, arg_b16x4, arg_b16x8, 4); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vbfdotq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 0);
	vbfdotq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 1);
	vbfdotq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vbfdotq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_vector_multiply_accumulate_by_scalar_f32(bfloat16x8_t arg_b16x8, bfloat16x4_t arg_b16x4, float32x4_t arg_f32x4) {
	vbfmlalbq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 0);
	vbfmlalbq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 3);
	vbfmlalbq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vbfmlalbq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vbfmlalbq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 0);
	vbfmlalbq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 7);
	vbfmlalbq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vbfmlalbq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vbfmlaltq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 0);
	vbfmlaltq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 3);
	vbfmlaltq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vbfmlaltq_lane_f32(arg_f32x4, arg_b16x8, arg_b16x4, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

	vbfmlaltq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 0);
	vbfmlaltq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 7);
	vbfmlaltq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
	vbfmlaltq_laneq_f32(arg_f32x4, arg_b16x8, arg_b16x8, 8); // expected-error-re +{{argument value {{.*}} is outside the valid range}}

}

