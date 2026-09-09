// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -fsyntax-only -verify

// Tests compile-time semantic errors for ACE v1 intrinsics

#include <immintrin.h>

unsigned char TILE_ID = 2;
unsigned char SCALE_GROUP = 0;

void test_tile_range_errors(__m512bh bf, __m512i i) {
  // Tile index must be in range [0, 7]
  _tile_ace_zero(8);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_zero(16); // expected-error {{argument value 16 is outside the valid range [0, 7]}}

  // Tile insert index errors
  _tile_insertcol(8, i, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_insertrow(16, i, 0); // expected-error {{argument value 16 is outside the valid range [0, 7]}}

  // Outer product tile index errors
  _tile_op4buud_epi32(8, i, i);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op4busd_epi32(8, i, i);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op4bssd_epi32(16, i, i); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_op4bsud_epi32(8, i, i);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op2bf16_ps(9, bf, bf); // expected-error {{argument value 9 is outside the valid range [0, 7]}}

  // Mixed precision outer product tile index errors
  _tile_op4mxhf8_ps(8, i, i, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op4mxbhf8_ps(8, i, i, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op4mxhbf8_ps(8, i, i, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op4mxbf8_ps(8, i, i, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_op4mxbss_ps(8, i, i, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Row read tile index errors
  _tile_cvtrow_epi32_ps(8, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_cvtrowh_ps_pbh(8, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_cvtrowl_ps_pbh(8, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_cvtrowh_ps_ph(8, 0);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_cvtrowl_ps_ph(8, 0);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_extractrow(8, 0);       // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_id_constness(__m512bh bf, __m512i i) {
  // The tile index must be an integer constant expression.
  _tile_ace_zero(TILE_ID); // expected-error {{argument to '__builtin_ia32_tilezero' must be a constant integer}}

  _tile_insertcol(TILE_ID, i, 0); // expected-error {{argument to '__builtin_ia32_tilemovcolinsert' must be a constant integer}}
  _tile_insertrow(TILE_ID, i, 0); // expected-error {{argument to '__builtin_ia32_tilemovrowinsert' must be a constant integer}}

  _tile_op4buud_epi32(TILE_ID, i, i);  // expected-error {{argument to '__builtin_ia32_top4buud' must be a constant integer}}
  _tile_op4busd_epi32(TILE_ID, i, i);  // expected-error {{argument to '__builtin_ia32_top4busd' must be a constant integer}}
  _tile_op4bssd_epi32(TILE_ID, i, i);  // expected-error {{argument to '__builtin_ia32_top4bssd' must be a constant integer}}
  _tile_op4bsud_epi32(TILE_ID, i, i);  // expected-error {{argument to '__builtin_ia32_top4bsud' must be a constant integer}}
  _tile_op2bf16_ps(TILE_ID, bf, bf); // expected-error {{argument to '__builtin_ia32_top2bf16ps' must be a constant integer}}

  _tile_op4mxhf8_ps(TILE_ID, i, i, 0);  // expected-error {{argument to '__builtin_ia32_top4mxhf8ps' must be a constant integer}}
  _tile_op4mxbhf8_ps(TILE_ID, i, i, 0); // expected-error {{argument to '__builtin_ia32_top4mxbhf8ps' must be a constant integer}}
  _tile_op4mxhbf8_ps(TILE_ID, i, i, 0); // expected-error {{argument to '__builtin_ia32_top4mxhbf8ps' must be a constant integer}}
  _tile_op4mxbf8_ps(TILE_ID, i, i, 0);  // expected-error {{argument to '__builtin_ia32_top4mxbf8ps' must be a constant integer}}
  _tile_op4mxbss_ps(TILE_ID, i, i, 0);  // expected-error {{argument to '__builtin_ia32_top4mxbssps' must be a constant integer}}

  _tile_cvtrow_epi32_ps(TILE_ID, 0); // expected-error {{argument to '__builtin_ia32_tcvtrowd2ps' must be a constant integer}}
  _tile_cvtrowh_ps_pbh(TILE_ID, 0);  // expected-error {{argument to '__builtin_ia32_tcvtrowps2bf16h' must be a constant integer}}
  _tile_cvtrowl_ps_pbh(TILE_ID, 0);  // expected-error {{argument to '__builtin_ia32_tcvtrowps2bf16l' must be a constant integer}}
  _tile_cvtrowh_ps_ph(TILE_ID, 0);   // expected-error {{argument to '__builtin_ia32_tcvtrowps2phh' must be a constant integer}}
  _tile_cvtrowl_ps_ph(TILE_ID, 0);   // expected-error {{argument to '__builtin_ia32_tcvtrowps2phl' must be a constant integer}}
  _tile_extractrow(TILE_ID, 0);      // expected-error {{argument to '__builtin_ia32_tilemovrow' must be a constant integer}}
}

void test_scale_group_errors(__m512i i) {
  // The scale group selector encodes the A group in bits [1:0] and the B group
  // in bits [4:3]; bits [7:6], [5] and [2] are reserved, leaving 16 of the 256
  // encodings valid.
  _tile_op4mxhf8_ps(0, i, i, 0x04);  // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  _tile_op4mxbhf8_ps(0, i, i, 0x20); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  _tile_op4mxhbf8_ps(0, i, i, 0x40); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  _tile_op4mxbf8_ps(0, i, i, 0x80);  // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  _tile_op4mxbss_ps(0, i, i, 0xFF);  // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}

  // -1 wraps to 0xFF, which has reserved bits set.
  _tile_op4mxbf8_ps(0, i, i, -1); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}

  // The tile index and the scale group are both checked.
  _tile_op4mxbf8_ps(8, i, i, 0x04); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_scale_group_constness(__m512i i) {
  // The scale group selector must also be an integer constant expression.
  _tile_op4mxhf8_ps(0, i, i, SCALE_GROUP);  // expected-error {{argument to '__builtin_ia32_top4mxhf8ps' must be a constant integer}}
  _tile_op4mxbhf8_ps(0, i, i, SCALE_GROUP); // expected-error {{argument to '__builtin_ia32_top4mxbhf8ps' must be a constant integer}}
  _tile_op4mxhbf8_ps(0, i, i, SCALE_GROUP); // expected-error {{argument to '__builtin_ia32_top4mxhbf8ps' must be a constant integer}}
  _tile_op4mxbf8_ps(0, i, i, SCALE_GROUP);  // expected-error {{argument to '__builtin_ia32_top4mxbf8ps' must be a constant integer}}
  _tile_op4mxbss_ps(0, i, i, SCALE_GROUP);  // expected-error {{argument to '__builtin_ia32_top4mxbssps' must be a constant integer}}
}

void test_scale_group_valid(__m512i i) {
  // All 16 legal encodings: A group in bits [1:0], B group in bits [4:3].
  _tile_op4mxbf8_ps(0, i, i, 0x00);
  _tile_op4mxbf8_ps(0, i, i, 0x03);
  _tile_op4mxbf8_ps(0, i, i, 0x18);
  _tile_op4mxbf8_ps(0, i, i, 0x1B);
}

void test_scale_group_helpers(__m512i i) {
  // The helper macros only ever set the two group fields, so every
  // combination of in-range group numbers is accepted.
  _tile_op4mxbf8_ps(0, i, i, _MM_ACE_SCALE_A(0) | _MM_ACE_SCALE_B(0));
  _tile_op4mxbf8_ps(0, i, i, _MM_ACE_SCALE_A(3) | _MM_ACE_SCALE_B(3));
  _tile_op4mxbss_ps(0, i, i, _MM_ACE_SCALE_A(1) | _MM_ACE_SCALE_B(2));

  // Either field alone is valid; the other group defaults to 0.
  _tile_op4mxbf8_ps(0, i, i, _MM_ACE_SCALE_A(2));
  _tile_op4mxbf8_ps(0, i, i, _MM_ACE_SCALE_B(2));

  // The macros deliberately do not mask their argument, so an out-of-range
  // group number spills into a reserved bit and is diagnosed rather than
  // silently truncated.
  _tile_op4mxbf8_ps(0, i, i, _MM_ACE_SCALE_A(4)); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  _tile_op4mxbf8_ps(0, i, i, _MM_ACE_SCALE_B(4)); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
}
