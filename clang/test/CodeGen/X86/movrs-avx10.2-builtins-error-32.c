// RUN: %clang_cc1 -ffreestanding %s -Wno-implicit-function-declaration -triple=i386-unknown-unknown -target-feature +movrs -target-feature +avx10.2-256 -verify

#include <immintrin.h>
__m128i test_mm_loadrs_epi8(const __m128i * __A) {
  return _mm_loadrs_epi8(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_mask_loadrs_epi8(__m128i __A, __mmask16 __B, const __m128i * __C) {
  return _mm_mask_loadrs_epi8(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_maskz_loadrs_epi8(__mmask16 __A, const __m128i * __B) {
  return _mm_maskz_loadrs_epi8(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m256i test_mm256_loadrs_epi8(const __m256i * __A) {
  return _mm256_loadrs_epi8(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_mask_loadrs_epi8(__m256i __A, __mmask32 __B, const __m256i * __C) {
  return _mm256_mask_loadrs_epi8(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_maskz_loadrs_epi8(__mmask32 __A, const __m256i * __B) {
  return _mm256_maskz_loadrs_epi8(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m128i test_mm_loadrs_epi32(const __m128i * __A) {
  return _mm_loadrs_epi32(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_mask_loadrs_epi32(__m128i __A, __mmask8 __B, const __m128i * __C) {
  return _mm_mask_loadrs_epi32(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_maskz_loadrs_epi32(__mmask8 __A, const __m128i * __B) {
  return _mm_maskz_loadrs_epi32(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m256i test_mm256_loadrs_epi32(const __m256i * __A) {
  return _mm256_loadrs_epi32(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_mask_loadrs_epi32(__m256i __A, __mmask8 __B, const __m256i * __C) {
  return _mm256_mask_loadrs_epi32(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_maskz_loadrs_epi32(__mmask8 __A, const __m256i * __B) {
  return _mm256_maskz_loadrs_epi32(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m128i test_mm_loadrs_epi64(const __m128i * __A) {
  return _mm_loadrs_epi64(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_mask_loadrs_epi64(__m128i __A, __mmask8 __B, const __m128i * __C) {
  return _mm_mask_loadrs_epi64(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_maskz_loadrs_epi64(__mmask8 __A, const __m128i * __B) {
  return _mm_maskz_loadrs_epi64(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m256i test_mm256_loadrs_epi64(const __m256i * __A) {
  return _mm256_loadrs_epi64(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_mask_loadrs_epi64(__m256i __A, __mmask8 __B, const __m256i * __C) {
  return _mm256_mask_loadrs_epi64(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_maskz_loadrs_epi64(__mmask8 __A, const __m256i * __B) {
  return _mm256_maskz_loadrs_epi64(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m128i test_mm_loadrs_epi16(const __m128i * __A) {
  return _mm_loadrs_epi16(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_mask_loadrs_epi16(__m128i __A, __mmask8 __B, const __m128i * __C) {
  return _mm_mask_loadrs_epi16(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m128i test_mm_maskz_loadrs_epi16(__mmask8 __A, const __m128i * __B) {
  return _mm_maskz_loadrs_epi16(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m128i' (vector of 2 'long long' values)}}
}

__m256i test_mm256_loadrs_epi16(const __m256i * __A) {
  return _mm256_loadrs_epi16(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_mask_loadrs_epi16(__m256i __A, __mmask16 __B, const __m256i * __C) {
  return _mm256_mask_loadrs_epi16(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}

__m256i test_mm256_maskz_loadrs_epi16(__mmask16 __A, const __m256i * __B) {
  return _mm256_maskz_loadrs_epi16(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m256i' (vector of 4 'long long' values)}}
}
