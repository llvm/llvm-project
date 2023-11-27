// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sve2 -target-feature +sme2 -target-feature +sme-i16i64 -target-feature +sme-f64f64 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme_draft_spec_subject_to_change.h>

void test_outer_product(svbool_t pred, svint16_t s16, svuint16_t u16, svint32_t s32, svuint32_t u32) __arm_streaming __arm_shared_za {
  // Test Tile Range
  svmopa_za32_u16_m(4, pred, pred, u16, u16); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svmopa_za32_s16_m(4, pred, pred, s16, s16); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svmops_za32_u16_m(4, pred, pred, u16, u16); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svmops_za32_s16_m(4, pred, pred, s16, s16); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svbmopa_za32_u32_m(4, pred, pred, u32, u32); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svbmopa_za32_s32_m(4, pred, pred, s32, s32); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svbmops_za32_u32_m(4, pred, pred, u32, u32); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svbmops_za32_s32_m(4, pred, pred, s32, s32); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_ldr_str_zt(const void *const_base, void *base) __arm_streaming_compatible __arm_shared_za __arm_preserves_za {
  svldr_zt(1, const_base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svstr_zt(1, base);       // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}
// REQUIRES: aarch64-registered-target
#include <arm_sme_draft_spec_subject_to_change.h>

void test_svluti2_lane_zt_x4(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32, svfloat16_t zn_f16, svbfloat16_t zn_bf16, svfloat32_t zn_f32) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti2_lane_zt_u8_x4(1, zn_u8, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8_x4(0, zn_u8, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_u16_x4(1, zn_u16, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16_x4(0, zn_u16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_u32_x4(1, zn_u32, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32_x4(0, zn_u32, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svluti2_lane_zt_f16_x4(1, zn_f16, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svluti2_lane_zt_f16_x4(0, zn_f16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svluti2_lane_zt_bf16_x4(1, zn_bf16, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svluti2_lane_zt_bf16_x4(0, zn_bf16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svluti2_lane_zt_f32_x4(1, zn_f32, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svluti2_lane_zt_f32_x4(0, zn_f32, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_svluti4_lane_zt_x4(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32, svfloat16_t zn_f16, svbfloat16_t zn_bf16, svfloat32_t zn_f32) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti4_lane_zt_u16_x4(1, zn_u16, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16_x4(0, zn_u16, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  // Test Reg Offset
  svluti4_lane_zt_u32_x4(1, zn_u32, 1);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32_x4(0, zn_u32, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svluti4_lane_zt_f16_x4(1, zn_f16, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svluti4_lane_zt_f16_x4(0, zn_f16, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svluti4_lane_zt_bf16_x4(1, zn_bf16, 0); // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svluti4_lane_zt_bf16_x4(0, zn_bf16, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svluti4_lane_zt_f32_x4(1, zn_f32, 1);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svluti4_lane_zt_f32_x4(0, zn_f32, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}

