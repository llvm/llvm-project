// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -fsyntax-only -verify
// RUN: %clang_cc1 -x c++ %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
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
  _tile_ace_cvtrow_epi32_ps(8, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_cvtrowh_ps_pbh(8, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_cvtrowl_ps_pbh(8, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_cvtrowh_ps_ph(8, 0);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_cvtrowl_ps_ph(8, 0);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_ace_extractrow(8, 0);       // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_row_index_range(__m512i i, unsigned idx) {
  // Only the low four bits of the index name a row, so one past the last row
  // is as wrong as a wildly large index.
  _tile_insertcol(0, i, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  _tile_insertrow(0, i, 999); // expected-error {{argument value 999 is outside the valid range [0, 15]}}
  _tile_ace_extractrow(0, 300);      // expected-error {{argument value 300 is outside the valid range [0, 15]}}
  // The index parameter is unsigned, so a negative one is reported as the
  // value it converts to.
  _tile_ace_cvtrow_epi32_ps(0, -1);  // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  _tile_ace_cvtrowh_ps_pbh(0, 20);   // expected-error {{argument value 20 is outside the valid range [0, 15]}}
  _tile_ace_cvtrowl_ps_pbh(0, 256);  // expected-error {{argument value 256 is outside the valid range [0, 15]}}
  _tile_ace_cvtrowh_ps_ph(0, 256);   // expected-error {{argument value 256 is outside the valid range [0, 15]}}
  _tile_ace_cvtrowl_ps_ph(0, 256);   // expected-error {{argument value 256 is outside the valid range [0, 15]}}

  // An index is folded before it is checked.
  _tile_ace_extractrow(0, 8 + 8); // expected-error {{argument value 16 is outside the valid range [0, 15]}}

  // An index that is in range, or not known at all, has to stay silent.
  _tile_insertrow(0, i, 15);
  _tile_insertrow(0, i, idx);
  _tile_ace_extractrow(0, 0);
  _tile_ace_extractrow(0, idx);
  _tile_ace_extractrow(0, 2 * 7);
}

void test_tile_id_constness(__m512bh bf, __m512i i) {
  // The tile index must be an integer constant expression.
  _tile_ace_zero(TILE_ID); // expected-error {{argument to '__builtin_ia32_acev1_tilezero' must be a constant integer}}

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

  _tile_ace_cvtrow_epi32_ps(TILE_ID, 0); // expected-error {{argument to '__builtin_ia32_acev1_tcvtrowd2ps' must be a constant integer}}
  _tile_ace_cvtrowh_ps_pbh(TILE_ID, 0);  // expected-error {{argument to '__builtin_ia32_acev1_tcvtrowps2bf16h' must be a constant integer}}
  _tile_ace_cvtrowl_ps_pbh(TILE_ID, 0);  // expected-error {{argument to '__builtin_ia32_acev1_tcvtrowps2bf16l' must be a constant integer}}
  _tile_ace_cvtrowh_ps_ph(TILE_ID, 0);   // expected-error {{argument to '__builtin_ia32_acev1_tcvtrowps2phh' must be a constant integer}}
  _tile_ace_cvtrowl_ps_ph(TILE_ID, 0);   // expected-error {{argument to '__builtin_ia32_acev1_tcvtrowps2phl' must be a constant integer}}
  _tile_ace_extractrow(TILE_ID, 0);      // expected-error {{argument to '__builtin_ia32_acev1_tilemovrowextract' must be a constant integer}}
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

void test_scale_group_struct_api(__acetile *t, __m512i i, unsigned char s) {
  // The struct API uses the internal builtins, where the group is argument 3.
  __tile_ace_top4mxhf8ps(t, i, i, 0x04);  // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  __tile_ace_top4mxbhf8ps(t, i, i, 0x20); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  __tile_ace_top4mxhbf8ps(t, i, i, 0x40); // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  __tile_ace_top4mxbf8ps(t, i, i, 0x80);  // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}
  __tile_ace_top4mxbssps(t, i, i, 0xFF);  // expected-error {{scale group argument must have bits 2, 5, 6, and 7 clear}}

  // The group is immarg, so a run-time one used to reach the IR verifier.
  __tile_ace_top4mxbf8ps(t, i, i, s); // expected-error {{argument to '__builtin_ia32_top4mxbf8ps_internal' must be a constant integer}}

  __tile_ace_top4mxbf8ps(t, i, i, _MM_ACE_SCALE_A(1) | _MM_ACE_SCALE_B(2));
}

#ifdef __cplusplus
// A dependent index is only known once the template is instantiated.
template <int N> void test_dependent_row_index() {
  _tile_ace_extractrow(0, N); // expected-error {{argument value 999 is outside the valid range [0, 15]}}
}

void test_dependent_row_index_instantiations() {
  test_dependent_row_index<999>(); // expected-note {{in instantiation of function template specialization 'test_dependent_row_index<999>' requested here}}
  test_dependent_row_index<15>();
}
#endif
