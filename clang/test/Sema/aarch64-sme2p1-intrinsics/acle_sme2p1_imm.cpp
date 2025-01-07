// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sme -target-feature +sme2p1 -target-feature +bf16 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void tests_readz_tile_to_vector_single(uint32_t slice) __arm_streaming __arm_inout("za") {
  svreadz_hor_za8_s8(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 0]}}
  svreadz_hor_za16_s16(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svreadz_hor_za32_s32(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
  svreadz_hor_za64_s64(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 7]}}
  svreadz_hor_za128_s8(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  svreadz_hor_za128_s16(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  svreadz_hor_za128_s32(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  svreadz_hor_za128_s64(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  svreadz_hor_za128_bf16(-1, slice); // expected-error {{argument value 18446744073709551615 is outside the valid range [0, 15]}}
  return;
}


