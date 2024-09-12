// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-unknown-unknown -target-feature +avx10.2-256 -emit-llvm -Wall -Werror -verify

unsigned long long test_mm_cvttssd(unsigned long long __A) {
  return _mm_cvttssd(__A); // expected-error {{call to undeclared function '_mm_cvttssd'}}
}

unsigned long long test_mm_cvttsss(unsigned long long __A) {
  return _mm_cvttsss(__A); // expected-error {{call to undeclared function '_mm_cvttsss'}}
}

#include <immintrin.h>
#include <stddef.h>

__m128i test_mm256_cvtts_roundpd_epi32(__m256d A) {
  return _mm256_cvtts_roundpd_epi32(A, 22); // expected-error {{invalid rounding argument}}
}
__m128i test_mm256_mask_cvtts_roundpd_epi32(__m128i W, __mmask8 U, __m256d A) {
  return _mm256_mask_cvtts_roundpd_epi32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m128i test_mm256_maskz_cvtts_roundpd_epi32(__mmask8 U, __m256d A) {
  return _mm256_maskz_cvtts_roundpd_epi32(U, A, 22); // expected-error {{invalid rounding argument}}
}

__m128i test_mm256_cvtts_roundpd_epu32(__m256d A) {
  return _mm256_cvtts_roundpd_epu32(A, 22); // expected-error {{invalid rounding argument}}
}
__m128i test_mm256_mask_cvtts_roundpd_epu32(__m128i W, __mmask8 U, __m256d A) {
  return _mm256_mask_cvtts_roundpd_epu32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m128i test_mm256_maskz_cvtts_roundpd_epu32(__mmask8 U, __m256d A) {
  return _mm256_maskz_cvtts_roundpd_epu32(U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_cvtts_roundps_epi32(__m256 A) {
  return _mm256_cvtts_roundps_epi32(A, 22); // expected-error {{invalid rounding argument}}
}
__m256i test_mm256_mask_cvtts_roundps_epi32(__m256i W, __mmask8 U, __m256 A) {
  return _mm256_mask_cvtts_roundps_epi32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_cvtts_roundps_epi32(__mmask8 U, __m256 A) {
  return _mm256_maskz_cvtts_roundps_epi32(U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_cvtts_roundps_epu32(__m256 A) {
  return _mm256_cvtts_roundps_epu32(A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_cvtts_roundps_epu32(__m256i W, __mmask8 U, __m256 A) {
  return _mm256_mask_cvtts_roundps_epu32(W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_cvtts_roundps_epu32(__mmask8 U, __m256 A) {
  return _mm256_maskz_cvtts_roundps_epu32(U, A, 22); // expected-error {{invalid rounding argument}}
}
