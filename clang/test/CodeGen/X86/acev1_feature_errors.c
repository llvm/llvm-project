// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +avx10.1 -emit-llvm -o /dev/null -verify=nofeature
// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o /dev/null -verify=feature

// Tests that the ACE v1 intrinsics require the acev1 target feature.

// feature-no-diagnostics

#include <immintrin.h>

void test_needs_acev1(__m512bh bf, __m512i i) {
  _tile_insertcol(0, i, 0); // nofeature-error {{'__builtin_ia32_tilemovcolinsert' needs target feature acev1}}
  _tile_insertrow(0, i, 0); // nofeature-error {{'__builtin_ia32_tilemovrowinsert' needs target feature acev1}}
  _tile_op4buud_epi32(0, i, i); // nofeature-error {{'__builtin_ia32_top4buud' needs target feature acev1}}
  _tile_op2bf16_ps(0, bf, bf); // nofeature-error {{'__builtin_ia32_top2bf16ps' needs target feature acev1}}
  _tile_op4mxbf8_ps(0, i, i, 0); // nofeature-error {{'__builtin_ia32_top4mxbf8ps' needs target feature acev1}}

  _tile_ace_zero(0); // nofeature-error {{'__builtin_ia32_tilezero' needs target feature amx-tile|acev1}}
  _tile_cvtrow_epi32_ps(0, 0); // nofeature-error {{'__builtin_ia32_tcvtrowd2ps' needs target feature amx-avx512,avx10.2|acev1}}
  _tile_extractrow(0, 0); // nofeature-error {{'__builtin_ia32_tilemovrow' needs target feature amx-avx512,avx10.2|acev1}}

  _tile_ace_loadconfig(&i); // nofeature-error {{always_inline function '_tile_ace_loadconfig' requires target feature 'acev1'}}
  _tile_ace_storeconfig(&i); // nofeature-error {{always_inline function '_tile_ace_storeconfig' requires target feature 'acev1'}}
  _tile_ace_release(); // nofeature-error {{always_inline function '_tile_ace_release' requires target feature 'acev1'}}
  _bsr0_init(); // nofeature-error {{always_inline function '_bsr0_init' requires target feature 'acev1'}}
  _bsr0_insertfull(i, i); // nofeature-error {{always_inline function '_bsr0_insertfull' requires target feature 'acev1'}}
  _bsr0_inserth(i); // nofeature-error {{always_inline function '_bsr0_inserth' requires target feature 'acev1'}}
  _bsr0_insertl(i); // nofeature-error {{always_inline function '_bsr0_insertl' requires target feature 'acev1'}}
}
