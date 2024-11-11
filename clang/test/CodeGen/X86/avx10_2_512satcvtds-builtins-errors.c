// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-unknown-unknown -target-feature +avx10.2-512 -Wall -Werror -verify

#include <immintrin.h>
#include <stddef.h>

__m256i test_mm512_cvtts_roundpd_epi32(__m512d A) {
  return _mm512_cvtts_roundpd_epi32(A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm512_mask_cvtts_roundpd_epi32(__m256i W, __mmask8 U, __m512d A) {
  return _mm512_mask_cvtts_roundpd_epi32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm512_maskz_cvtts_roundpd_epi32(__mmask8 U, __m512d A) {
  return _mm512_maskz_cvtts_roundpd_epi32(U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm512_cvtts_roundpd_epu32(__m512d A) {
  return _mm512_cvtts_roundpd_epu32(A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm512_mask_cvtts_roundpd_epu32(__m256i W, __mmask8 U, __m512d A) {
  return _mm512_mask_cvtts_roundpd_epu32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm512_maskz_cvtts_roundpd_epu32(__mmask8 U, __m512d A) {
  return _mm512_maskz_cvtts_roundpd_epu32(U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_cvtts_roundps_epi32(__m512 A) {
  return _mm512_cvtts_roundps_epi32(A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_cvtts_roundps_epi32(__m512i W, __mmask8 U, __m512 A) {
  return _mm512_mask_cvtts_roundps_epi32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_cvtts_roundps_epi32(__mmask8 U, __m512 A) {
  return _mm512_maskz_cvtts_roundps_epi32(U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_cvtts_roundps_epu32(__m512 A) {
  return _mm512_cvtts_roundps_epu32(A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_cvtts_roundps_epu32(__m512i W, __mmask8 U, __m512 A) {
  return _mm512_mask_cvtts_roundps_epu32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_cvtts_roundps_epu32(__mmask8 U, __m512 A) {
  return _mm512_maskz_cvtts_roundps_epu32(U, A, 22); // expected-error {{invalid rounding argument}}
}