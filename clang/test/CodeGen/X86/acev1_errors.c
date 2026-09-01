// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -fsyntax-only -verify

// Tests compile-time semantic errors for ACE v1 intrinsics

#include <immintrin.h>

void test_tile_range_errors(__m512bh bf, __m512i i) {
  // Tile index must be in range [0, 7]
  _tile_ace_zero(8);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_zero(16); // expected-error {{argument value 16 is outside the valid range [0, 7]}}

  // Outer product tile index errors
  _tile_top4buud(8, i, i);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_top4bssd(16, i, i); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_top2bf16ps(9, bf, bf); // expected-error {{argument value 9 is outside the valid range [0, 7]}}
}
