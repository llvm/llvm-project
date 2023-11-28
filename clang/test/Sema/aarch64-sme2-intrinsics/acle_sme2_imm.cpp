// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sve2 -target-feature +sme2 -target-feature +sve -fsyntax-only -verify %s

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
