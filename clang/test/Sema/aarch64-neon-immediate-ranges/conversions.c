// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon  -ffreestanding -fsyntax-only -verify %s

#include <arm_neon.h>
// REQUIRES: aarch64-registered-target


void test_conversions_f32(float32_t arg_f32, float32x2_t arg_f32x2, float32x4_t arg_f32x4) {
	vcvt_n_s32_f32(arg_f32x2, 1);
	vcvt_n_s32_f32(arg_f32x2, 32);
	vcvt_n_s32_f32(arg_f32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_s32_f32(arg_f32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_s32_f32(arg_f32x4, 1);
	vcvtq_n_s32_f32(arg_f32x4, 32);
	vcvtq_n_s32_f32(arg_f32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_s32_f32(arg_f32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvt_n_u32_f32(arg_f32x2, 1);
	vcvt_n_u32_f32(arg_f32x2, 32);
	vcvt_n_u32_f32(arg_f32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_u32_f32(arg_f32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_u32_f32(arg_f32x4, 1);
	vcvtq_n_u32_f32(arg_f32x4, 32);
	vcvtq_n_u32_f32(arg_f32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_u32_f32(arg_f32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvts_n_s32_f32(arg_f32, 1);
	vcvts_n_s32_f32(arg_f32, 32);
	vcvts_n_s32_f32(arg_f32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvts_n_s32_f32(arg_f32, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvts_n_u32_f32(arg_f32, 1);
	vcvts_n_u32_f32(arg_f32, 32);
	vcvts_n_u32_f32(arg_f32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvts_n_u32_f32(arg_f32, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_f64(float64x1_t arg_f64x1, float64x2_t arg_f64x2, float64_t arg_f64) {
	vcvt_n_s64_f64(arg_f64x1, 1);
	vcvt_n_s64_f64(arg_f64x1, 64);
	vcvt_n_s64_f64(arg_f64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_s64_f64(arg_f64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_s64_f64(arg_f64x2, 1);
	vcvtq_n_s64_f64(arg_f64x2, 64);
	vcvtq_n_s64_f64(arg_f64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_s64_f64(arg_f64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvt_n_u64_f64(arg_f64x1, 1);
	vcvt_n_u64_f64(arg_f64x1, 64);
	vcvt_n_u64_f64(arg_f64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_u64_f64(arg_f64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_u64_f64(arg_f64x2, 1);
	vcvtq_n_u64_f64(arg_f64x2, 64);
	vcvtq_n_u64_f64(arg_f64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_u64_f64(arg_f64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtd_n_s64_f64(arg_f64, 1);
	vcvtd_n_s64_f64(arg_f64, 64);
	vcvtd_n_s64_f64(arg_f64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtd_n_s64_f64(arg_f64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtd_n_u64_f64(arg_f64, 1);
	vcvtd_n_u64_f64(arg_f64, 64);
	vcvtd_n_u64_f64(arg_f64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtd_n_u64_f64(arg_f64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_s32(int32_t arg_i32, int32x2_t arg_i32x2, int32x4_t arg_i32x4) {
	vcvt_n_f32_s32(arg_i32x2, 1);
	vcvt_n_f32_s32(arg_i32x2, 32);
	vcvt_n_f32_s32(arg_i32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_f32_s32(arg_i32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_f32_s32(arg_i32x4, 1);
	vcvtq_n_f32_s32(arg_i32x4, 32);
	vcvtq_n_f32_s32(arg_i32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_f32_s32(arg_i32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvts_n_f32_s32(arg_i32, 1);
	vcvts_n_f32_s32(arg_i32, 32);
	vcvts_n_f32_s32(arg_i32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvts_n_f32_s32(arg_i32, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_u32(uint32x4_t arg_u32x4, uint32x2_t arg_u32x2, uint32_t arg_u32) {
	vcvt_n_f32_u32(arg_u32x2, 1);
	vcvt_n_f32_u32(arg_u32x2, 32);
	vcvt_n_f32_u32(arg_u32x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_f32_u32(arg_u32x2, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_f32_u32(arg_u32x4, 1);
	vcvtq_n_f32_u32(arg_u32x4, 32);
	vcvtq_n_f32_u32(arg_u32x4, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_f32_u32(arg_u32x4, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvts_n_f32_u32(arg_u32, 1);
	vcvts_n_f32_u32(arg_u32, 32);
	vcvts_n_f32_u32(arg_u32, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvts_n_f32_u32(arg_u32, 33); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_s64(int64x2_t arg_i64x2, int64x1_t arg_i64x1, int64_t arg_i64) {
	vcvt_n_f64_s64(arg_i64x1, 1);
	vcvt_n_f64_s64(arg_i64x1, 64);
	vcvt_n_f64_s64(arg_i64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_f64_s64(arg_i64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_f64_s64(arg_i64x2, 1);
	vcvtq_n_f64_s64(arg_i64x2, 64);
	vcvtq_n_f64_s64(arg_i64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_f64_s64(arg_i64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtd_n_f64_s64(arg_i64, 1);
	vcvtd_n_f64_s64(arg_i64, 64);
	vcvtd_n_f64_s64(arg_i64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtd_n_f64_s64(arg_i64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

void test_conversions_u64(uint64x2_t arg_u64x2, uint64_t arg_u64, uint64x1_t arg_u64x1) {
	vcvt_n_f64_u64(arg_u64x1, 1);
	vcvt_n_f64_u64(arg_u64x1, 64);
	vcvt_n_f64_u64(arg_u64x1, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvt_n_f64_u64(arg_u64x1, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtq_n_f64_u64(arg_u64x2, 1);
	vcvtq_n_f64_u64(arg_u64x2, 64);
	vcvtq_n_f64_u64(arg_u64x2, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtq_n_f64_u64(arg_u64x2, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

	vcvtd_n_f64_u64(arg_u64, 1);
	vcvtd_n_f64_u64(arg_u64, 64);
	vcvtd_n_f64_u64(arg_u64, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
	vcvtd_n_f64_u64(arg_u64, 65); // expected-error-re {{argument value {{.*}} is outside the valid range}}

}

