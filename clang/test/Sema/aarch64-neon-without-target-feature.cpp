// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +dotprod  -target-feature +fullfp16 -target-feature +fp16fml -target-feature +i8mm -target-feature +bf16 -verify -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

// This test is testing the diagnostics that Clang emits when compiling without '+neon'.

#include <arm_neon.h>

void undefined(uint32x2_t v2i32, uint32x4_t v4i32, uint16x8_t v8i16, uint8x16_t v16i8, uint8x8_t v8i8, float32x2_t v2f32, float32x4_t v4f32, float16x4_t v4f16, float64x2_t v2f64, bfloat16x4_t v4bf16, __bf16 bf16, poly64_t poly64, poly64x2_t poly64x2) {
  // dotprod
  vdot_u32(v2i32, v8i8, v8i8); // expected-error {{always_inline function 'vdot_u32' requires target feature 'neon'}}
  vdot_laneq_u32(v2i32, v8i8, v16i8, 1); // expected-error {{always_inline function 'vdot_u32' requires target feature 'neon'}} expected-error {{'__builtin_neon_splat_laneq_v' needs target feature neon}}
  // fp16
  vceqz_f16(v4f16); // expected-error {{always_inline function 'vceqz_f16' requires target feature 'neon'}}
  vrnd_f16(v4f16); // expected-error {{always_inline function 'vrnd_f16' requires target feature 'neon'}}
  vmaxnm_f16(v4f16, v4f16); // expected-error {{always_inline function 'vmaxnm_f16' requires target feature 'neon'}}
  vrndi_f16(v4f16); // expected-error {{always_inline function 'vrndi_f16' requires target feature 'neon'}}
  // fp16fml depends on fp-armv8
  vfmlal_low_f16(v2f32, v4f16, v4f16); // expected-error {{always_inline function 'vfmlal_low_f16' requires target feature 'neon'}}
  // i8mm
  vmmlaq_s32(v4i32, v8i16, v8i16); // expected-error {{always_inline function 'vmmlaq_s32' requires target feature 'neon'}}
  vusdot_laneq_s32(v2i32, v8i8, v8i16, 0); // expected-error {{always_inline function 'vusdot_s32' requires target feature 'neon'}} expected-error {{'__builtin_neon_splat_laneq_v' needs target feature neon}}
  // bf16
  vbfdot_f32(v2f32, v4bf16, v4bf16); // expected-error {{always_inline function 'vbfdot_f32' requires target feature 'neon'}}
  vcreate_bf16(10);
  vdup_lane_bf16(v4bf16, 2); // expected-error {{'__builtin_neon_splat_lane_bf16' needs target feature bf16,neon}}
  vdup_n_bf16(bf16); // expected-error {{always_inline function 'vdup_n_bf16' requires target feature 'neon'}}
  vld1_bf16(0); // expected-error {{'__builtin_neon_vld1_bf16' needs target feature bf16,neon}}
  vcvt_f32_bf16(v4bf16); // expected-error {{always_inline function 'vcvt_f32_bf16' requires target feature 'neon'}}
  vcvt_bf16_f32(v4f32); // expected-error {{always_inline function 'vcvt_bf16_f32' requires target feature 'neon'}}
  vmull_p64(poly64, poly64);  // expected-error {{always_inline function 'vmull_p64' requires target feature 'neon'}}
  vmull_high_p64(poly64x2, poly64x2);  // expected-error {{always_inline function 'vmull_high_p64' requires target feature 'neon'}}
  vtrn1_s8(v8i8, v8i8); // expected-error {{always_inline function 'vtrn1_s8' requires target feature 'neon'}}
  vqabsq_s16(v8i16); // expected-error {{always_inline function 'vqabsq_s16' requires target feature 'neon'}}
  vbslq_s16(v8i16, v8i16, v8i16);// expected-error {{always_inline function 'vbslq_s16' requires target feature 'neon'}}
}
