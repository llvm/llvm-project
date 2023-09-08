// RUN: %clang_cc1 %s -ffreestanding -triple=i686-unknown-unknown -target-feature +sm3  -emit-llvm -fsyntax-only -verify

// XFAIL: *

#include <immintrin.h>

__m128i test_mm_sm3rnds2_epi32(__m128i __A, __m128i __B, __m128i __C) {
  return _mm_sm3rnds2_epi32(__A, __B, __C, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}
