// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx10v2aux -Wno-invalid-feature-combination -Wall -Werror -verify

#include <immintrin.h>

__m128i test_mm_unpack_epi8(__m128i __A) {
  return _mm_unpack_epi8(__A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m128i test_mm_mask_unpack_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  return _mm_mask_unpack_epi8(__W, __U, __A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m128i test_mm_maskz_unpack_epi8(__mmask16 __U, __m128i __A) {
  return _mm_maskz_unpack_epi8(__U, __A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m256i test_mm256_unpack_epi8(__m256i __A) {
  return _mm256_unpack_epi8(__A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m256i test_mm256_mask_unpack_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  return _mm256_mask_unpack_epi8(__W, __U, __A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m256i test_mm256_maskz_unpack_epi8(__mmask32 __U, __m256i __A) {
  return _mm256_maskz_unpack_epi8(__U, __A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m512i test_mm512_unpack_epi8(__m512i __A) {
  return _mm512_unpack_epi8(__A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m512i test_mm512_mask_unpack_epi8(__m512i __W, __mmask64 __U, __m512i __A) {
  return _mm512_mask_unpack_epi8(__W, __U, __A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}

__m512i test_mm512_maskz_unpack_epi8(__mmask64 __U, __m512i __A) {
  return _mm512_maskz_unpack_epi8(__U, __A, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
}
