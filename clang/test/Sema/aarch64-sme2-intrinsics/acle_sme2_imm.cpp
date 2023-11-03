// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sve2 -target-feature +sme2 -target-feature +sme-i16i64 -target-feature +sme-f64f64 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme_draft_spec_subject_to_change.h>

void test_multivector_read(uint32_t base) {

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
                            svuint64x2_t v64x2, svuint64x4_t v64x4) {

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

void test_ldr_str_zt(const void *const_base, void *base) __arm_streaming_compatible __arm_shared_za __arm_preserves_za {
  svldr_zt(1, const_base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svstr_zt(1, base);       // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}
