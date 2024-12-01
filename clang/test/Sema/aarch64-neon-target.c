// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -verify -emit-llvm -o - %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -target-feature +outline-atomics -verify -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

// Test that functions with the correct target attributes can use the correct NEON intrinsics.

#include <arm_neon.h>

__attribute__((target("dotprod")))
void dotprod(uint32x2_t v2i32, uint8x16_t v16i8, uint8x8_t v8i8) {
  vdot_u32(v2i32, v8i8, v8i8);
  vdot_laneq_u32(v2i32, v8i8, v16i8, 1);
}

__attribute__((target("fullfp16")))
void fp16(uint32x2_t v2i32, uint32x4_t v4i32, uint16x8_t v8i16, uint8x16_t v16i8, uint8x8_t v8i8, float32x2_t v2f32, float32x4_t v4f32, float16x4_t v4f16, bfloat16x4_t v4bf16) {
  vceqz_f16(v4f16);
  vrnd_f16(v4f16);
  vmaxnm_f16(v4f16, v4f16);
  vrndi_f16(v4f16);
}

__attribute__((target("fp16fml")))
void fp16fml(float32x2_t v2f32, float16x4_t v4f16) {
  vfmlal_low_f16(v2f32, v4f16, v4f16);
}

__attribute__((target("i8mm")))
void i8mm(uint32x2_t v2i32, uint32x4_t v4i32, uint16x8_t v8i16, uint8x16_t v16i8, uint8x8_t v8i8, float32x2_t v2f32, float32x4_t v4f32, float16x4_t v4f16, bfloat16x4_t v4bf16) {
  vmmlaq_s32(v4i32, v8i16, v8i16);
  vusdot_laneq_s32(v2i32, v8i8, v8i16, 0);
}

__attribute__((target("bf16")))
void bf16(uint32x2_t v2i32, uint32x4_t v4i32, uint16x8_t v8i16, uint8x16_t v16i8, uint8x8_t v8i8, float32x2_t v2f32, float32x4_t v4f32, float16x4_t v4f16, bfloat16x4_t v4bf16, __bf16 bf16) {
  vbfdot_f32(v2f32, v4bf16, v4bf16);
  vcreate_bf16(10);
  vdup_lane_bf16(v4bf16, 2);
  vdup_n_bf16(bf16);
  vld1_bf16(0);
  vcvt_f32_bf16(v4bf16);
  vcvt_bf16_f32(v4f32);
}

__attribute__((target("arch=armv8-a")))
uint64x2_t test_v8(uint64x2_t a, uint64x2_t b) {
  return veorq_u64(a, b);
}

__attribute__((target("arch=armv8.1-a")))
void test_v81(int32x2_t d, int32x4_t v, int s) {
  vqrdmlahq_s32(v, v, v);
  vqrdmlah_laneq_s32(d, d, v, 1);
  vqrdmlahh_s16(1, 1, 1);
}

__attribute__((target("arch=armv8.3-a+fp16")))
void test_v83(float32x4_t v4f32, float16x4_t v4f16, float64x2_t v2f64) {
  vcaddq_rot90_f32(v4f32, v4f32);
  vcmla_rot90_f16(v4f16, v4f16, v4f16);
  vcmlaq_rot270_f64(v2f64, v2f64, v2f64);
}

__attribute__((target("arch=armv8.5-a")))
void test_v85(float32x4_t v4f32) {
  vrnd32xq_f32(v4f32);
}

void undefined(uint32x2_t v2i32, uint32x4_t v4i32, uint16x8_t v8i16, uint8x16_t v16i8, uint8x8_t v8i8, float32x2_t v2f32, float32x4_t v4f32, float16x4_t v4f16, float64x2_t v2f64, bfloat16x4_t v4bf16, __bf16 bf16, poly64_t poly64, poly64x2_t poly64x2) {
  // dotprod
  vdot_u32(v2i32, v8i8, v8i8); // expected-error {{always_inline function 'vdot_u32' requires target feature 'dotprod'}}
  vdot_laneq_u32(v2i32, v8i8, v16i8, 1); // expected-error {{always_inline function 'vdot_u32' requires target feature 'dotprod'}}
  // fp16
  vceqz_f16(v4f16); // expected-error {{always_inline function 'vceqz_f16' requires target feature 'fullfp16'}}
  vrnd_f16(v4f16); // expected-error {{always_inline function 'vrnd_f16' requires target feature 'fullfp16'}}
  vmaxnm_f16(v4f16, v4f16); // expected-error {{always_inline function 'vmaxnm_f16' requires target feature 'fullfp16'}}
  vrndi_f16(v4f16); // expected-error {{always_inline function 'vrndi_f16' requires target feature 'fullfp16'}}
  // fp16fml depends on fp-armv8
  vfmlal_low_f16(v2f32, v4f16, v4f16); // expected-error {{always_inline function 'vfmlal_low_f16' requires target feature 'fp-armv8'}}
  // i8mm
  vmmlaq_s32(v4i32, v8i16, v8i16); // expected-error {{always_inline function 'vmmlaq_s32' requires target feature 'i8mm'}}
  vusdot_laneq_s32(v2i32, v8i8, v8i16, 0); // expected-error {{always_inline function 'vusdot_s32' requires target feature 'i8mm'}}
  // bf16
  vbfdot_f32(v2f32, v4bf16, v4bf16); // expected-error {{always_inline function 'vbfdot_f32' requires target feature 'bf16'}}
  vcreate_bf16(10);
  vdup_lane_bf16(v4bf16, 2); // expected-error {{'__builtin_neon_splat_lane_bf16' needs target feature bf16}}
  vdup_n_bf16(bf16); // expected-error {{always_inline function 'vdup_n_bf16' requires target feature 'bf16'}}
  vld1_bf16(0); // expected-error {{'__builtin_neon_vld1_bf16' needs target feature bf16}}
  vcvt_f32_bf16(v4bf16); // expected-error {{always_inline function 'vcvt_f32_bf16' requires target feature 'bf16'}}
  vcvt_bf16_f32(v4f32); // expected-error {{always_inline function 'vcvt_bf16_f32' requires target feature 'bf16'}}
  // v8.1 - qrdmla
  vqrdmlahq_s32(v4i32, v4i32, v4i32); // expected-error {{always_inline function 'vqrdmlahq_s32' requires target feature 'v8.1a'}}
  vqrdmlah_laneq_s32(v2i32, v2i32, v4i32, 1); // expected-error {{always_inline function 'vqrdmlah_s32' requires target feature 'v8.1a'}}
  vqrdmlahh_s16(1, 1, 1); // expected-error {{always_inline function 'vqrdmlahh_s16' requires target feature 'v8.1a'}}
  // 8.3 - complex
  vcaddq_rot90_f32(v4f32, v4f32); // expected-error {{always_inline function 'vcaddq_rot90_f32' requires target feature 'v8.3a'}}
  vcmla_rot90_f16(v4f16, v4f16, v4f16); // expected-error {{always_inline function 'vcmla_rot90_f16' requires target feature 'v8.3a'}}
  vcmlaq_rot270_f64(v2f64, v2f64, v2f64); // expected-error {{always_inline function 'vcmlaq_rot270_f64' requires target feature 'v8.3a'}}
  // 8.5 - frint
  vrnd32xq_f32(v4f32); // expected-error {{always_inline function 'vrnd32xq_f32' requires target feature 'v8.5a'}}

  vmull_p64(poly64, poly64);  // expected-error {{always_inline function 'vmull_p64' requires target feature 'aes'}}
  vmull_high_p64(poly64x2, poly64x2);  // expected-error {{always_inline function 'vmull_high_p64' requires target feature 'aes'}}

}
