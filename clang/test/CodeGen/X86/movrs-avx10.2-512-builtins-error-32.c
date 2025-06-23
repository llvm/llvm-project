// RUN: %clang_cc1 -ffreestanding %s -Wno-implicit-function-declaration -triple=i386-- -target-feature +movrs -target-feature +avx10.2-512 -verify

#include <immintrin.h>
__m512i test_mm512_loadrs_epi8(const __m512i * __A) {
  return _mm512_loadrs_epi8(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_mask_loadrs_epi8(__m512i __A, __mmask64 __B, const __m512i * __C) {
  return _mm512_mask_loadrs_epi8(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_maskz_loadrs_epi8(__mmask64 __A, const __m512i * __B) {
  return _mm512_maskz_loadrs_epi8(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_loadrs_epi32(const __m512i * __A) {
  return _mm512_loadrs_epi32(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_mask_loadrs_epi32(__m512i __A, __mmask16 __B, const __m512i * __C) {
  return _mm512_mask_loadrs_epi32(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_maskz_loadrs_epi32(__mmask16 __A, const __m512i * __B) {
  return _mm512_maskz_loadrs_epi32(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_loadrs_epi64(const __m512i * __A) {
  return _mm512_loadrs_epi64(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_mask_loadrs_epi64(__m512i __A, __mmask8 __B, const __m512i * __C) {
  return _mm512_mask_loadrs_epi64(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_maskz_loadrs_epi64(__mmask8 __A, const __m512i * __B) {
  return _mm512_maskz_loadrs_epi64(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_loadrs_epi16(const __m512i * __A) {
  return _mm512_loadrs_epi16(__A); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_mask_loadrs_epi16(__m512i __A, __mmask32 __B, const __m512i * __C) {
  return _mm512_mask_loadrs_epi16(__A, __B, __C); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}

__m512i test_mm512_maskz_loadrs_epi16(__mmask32 __A, const __m512i * __B) {
  return _mm512_maskz_loadrs_epi16(__A, __B); // expected-error {{returning 'int' from a function with incompatible result type '__m512i' (vector of 8 'long long' values)}}
}
