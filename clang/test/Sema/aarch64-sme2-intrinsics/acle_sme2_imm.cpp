// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sve2 -target-feature +sme2 -target-feature +sme-i16i64 -target-feature +sme-f64f64 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme_draft_spec_subject_to_change.h>

void test_multivector_read(uint32_t base) __arm_streaming __arm_shared_za __arm_preserves_za {

  // Test Tile Range
  svread_hor_za8_u8_vg2(1, base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svread_ver_za8_u8_vg2(1, base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svread_hor_za8_u8_vg4(1, base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svread_ver_za8_u8_vg4(1, base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svread_hor_za16_u16_vg2(2, base); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svread_ver_za16_u16_vg2(2, base); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svread_hor_za16_u16_vg4(2, base); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svread_ver_za16_u16_vg4(2, base); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svread_hor_za32_u32_vg2(4, base); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svread_ver_za32_u32_vg2(4, base); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svread_hor_za32_u32_vg4(4, base); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svread_ver_za32_u32_vg4(4, base); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svread_hor_za64_u64_vg2(8, base); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_ver_za64_u64_vg2(8, base); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_hor_za64_u64_vg4(8, base); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_ver_za64_u64_vg4(8, base); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_multivector_write(uint32_t base, svuint8x2_t v8x2, svuint8x4_t v8x4,
                            svuint16x2_t v16x2, svuint16x4_t v16x4,
                            svuint32x2_t v32x2, svuint32x4_t v32x4,
                            svuint64x2_t v64x2, svuint64x4_t v64x4) __arm_streaming __arm_shared_za {

  // Test Tile Range
  svwrite_hor_za8_u8_vg2(1, base, v8x2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_ver_za8_u8_vg2(1, base, v8x2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_hor_za8_u8_vg4(1, base, v8x4); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_ver_za8_u8_vg4(1, base, v8x4); // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svwrite_hor_za16_u16_vg2(2, base, v16x2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_ver_za16_u16_vg2(2, base, v16x2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_hor_za16_u16_vg4(2, base, v16x4); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_ver_za16_u16_vg4(2, base, v16x4); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svwrite_hor_za32_u32_vg2(4, base, v32x2); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_ver_za32_u32_vg2(4, base, v32x2); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_hor_za32_u32_vg4(4, base, v32x4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_ver_za32_u32_vg4(4, base, v32x4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svwrite_hor_za64_u64_vg2(8, base, v64x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_ver_za64_u64_vg2(8, base, v64x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_hor_za64_u64_vg4(8, base, v64x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_ver_za64_u64_vg4(8, base, v64x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

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

void test_ldr_zt(const void *const_base) __arm_streaming_compatible __arm_shared_za {
  svldr_zt(1, const_base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}

void test_str_zt(void *base) __arm_streaming_compatible __arm_shared_za __arm_preserves_za {
  svstr_zt(1, base);       // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}

void test_svluti2_lane_zt_x4(svuint8_t zn) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti2_lane_zt_u8_x4(1, zn, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8_x4(0, zn, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_u16_x4(1, zn, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16_x4(0, zn, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_u32_x4(1, zn, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32_x4(0, zn, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_f16_x4(1, zn, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_f16_x4(0, zn, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_bf16_x4(1, zn, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_bf16_x4(0, zn, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti2_lane_zt_f32_x4(1, zn, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_f32_x4(0, zn, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_svluti4_lane_zt_x4(svuint8_t zn) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti4_lane_zt_u16_x4(1, zn, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16_x4(0, zn, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  // Test Reg Offset
  svluti4_lane_zt_u32_x4(1, zn, 1);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32_x4(0, zn, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  // Test Reg Offset
  svluti4_lane_zt_f16_x4(1, zn, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_f16_x4(0, zn, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  // Test Reg Offset
  svluti4_lane_zt_bf16_x4(1, zn, 0); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_bf16_x4(0, zn, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  // Test Reg Offset
  svluti4_lane_zt_f32_x4(1, zn, 1);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_f32_x4(0, zn, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}

void test_svluti2_lane_zt(svuint8_t zn_u8) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti2_lane_zt_u8(1, zn_u8, 2);    // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8(0, zn_u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  // Test Reg Offset
  svluti2_lane_zt_u16(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16(0, zn_u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  // Test Reg Offset
  svluti2_lane_zt_f16(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_f16(0, zn_u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  // Test Reg Offset
  svluti2_lane_zt_bf16(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_bf16(0, zn_u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  // Test Reg Offset
  svluti2_lane_zt_u32(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32(0, zn_u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  // Test Reg Offset
  svluti2_lane_zt_f32(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_f32(0, zn_u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
}

void test_svluti4_lane_zt(svuint8_t zn_u8) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti4_lane_zt_u8(1, zn_u8, 2);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u8(0, zn_u8, 8);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti4_lane_zt_u16(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti4_lane_zt_f16(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_f16(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti4_lane_zt_bf16(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_bf16(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti4_lane_zt_u32(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti4_lane_zt_f32(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_f32(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_svluti2_lane_zt_x2(svuint8_t zn_u8) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti2_lane_zt_u8_x2(1, zn_u8, 2);    // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8_x2(0, zn_u8, 8);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti2_lane_zt_u16_x2(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16_x2(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti2_lane_zt_u32_x2(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32_x2(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti2_lane_zt_f16_x2(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_f16_x2(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  // Test Reg Offset
  svluti2_lane_zt_bf16_x2(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_bf16_x2(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
 // Test Reg Offset
  svluti2_lane_zt_f32_x2(1, zn_u8, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_f32_x2(0, zn_u8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_svluti4_lane_zt_x2(svuint8_t zn_u8) __arm_streaming __arm_shared_za __arm_preserves_za {
  // Test Reg Offset
  svluti4_lane_zt_u8_x2(1, zn_u8, 2);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u8_x2(0, zn_u8, 4);  // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti4_lane_zt_u16_x2(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16_x2(0, zn_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti4_lane_zt_u32_x2(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32_x2(0, zn_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti4_lane_zt_f16_x2(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_f16_x2(0, zn_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  // Test Reg Offset
  svluti4_lane_zt_bf16_x2(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_bf16_x2(0, zn_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
   // Test Reg Offset
  svluti4_lane_zt_f32_x2(1, zn_u8, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_f32_x2(0, zn_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_bfmlslb_bad_lane(svfloat32_t zda, svbfloat16_t zn, svbfloat16_t zm) __arm_streaming_compatible {
  svbfmlslb_lane_f32(zda, zn, zm, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svbfmlslt_lane_f32(zda, zn, zm, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}
