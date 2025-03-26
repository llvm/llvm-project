// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-512 \
// RUN: -Wall -Werror -verify
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-512 \
// RUN: -Wall -Werror -verify

#include <immintrin.h>

__m512i test_mm512_ipcvts_roundph_epi8(__m512h __A) {
  return _mm512_ipcvts_roundph_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvts_roundph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  return _mm512_mask_ipcvts_roundph_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvts_roundph_epi8(__mmask32 __A, __m512h __B) {
  return _mm512_maskz_ipcvts_roundph_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvts_roundph_epu8(__m512h __A) {
  return _mm512_ipcvts_roundph_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvts_roundph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  return _mm512_mask_ipcvts_roundph_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvts_roundph_epu8(__mmask32 __A, __m512h __B) {
  return _mm512_maskz_ipcvts_roundph_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvts_roundps_epi8(__m512 __A) {
  return _mm512_ipcvts_roundps_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvts_roundps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  return _mm512_mask_ipcvts_roundps_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvts_roundps_epi8(__mmask16 __A, __m512 __B) {
  return _mm512_maskz_ipcvts_roundps_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvts_roundps_epu8(__m512 __A) {
  return _mm512_ipcvts_roundps_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvts_roundps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  return _mm512_mask_ipcvts_roundps_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvts_roundps_epu8(__mmask16 __A, __m512 __B) {
  return _mm512_maskz_ipcvts_roundps_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvtts_roundph_epi8(__m512h __A) {
  return _mm512_ipcvtts_roundph_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvtts_roundph_epi8(__m512i __S, __mmask32 __A, __m512h __B) {
  return _mm512_mask_ipcvtts_roundph_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvtts_roundph_epi8(__mmask32 __A, __m512h __B) {
  return _mm512_maskz_ipcvtts_roundph_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvtts_roundph_epu8(__m512h __A) {
  return _mm512_ipcvtts_roundph_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvtts_roundph_epu8(__m512i __S, __mmask32 __A, __m512h __B) {
  return _mm512_mask_ipcvtts_roundph_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvtts_roundph_epu8(__mmask32 __A, __m512h __B) {
  return _mm512_maskz_ipcvtts_roundph_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvtts_roundps_epi8(__m512 __A) {
  return _mm512_ipcvtts_roundps_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvtts_roundps_epi8(__m512i __S, __mmask16 __A, __m512 __B) {
  return _mm512_mask_ipcvtts_roundps_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvtts_roundps_epi8(__mmask16 __A, __m512 __B) {
  return _mm512_maskz_ipcvtts_roundps_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_ipcvtts_roundps_epu8(__m512 __A) {
  return _mm512_ipcvtts_roundps_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_ipcvtts_roundps_epu8(__m512i __S, __mmask16 __A, __m512 __B) {
  return _mm512_mask_ipcvtts_roundps_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_ipcvtts_roundps_epu8(__mmask16 __A, __m512 __B) {
  return _mm512_maskz_ipcvtts_roundps_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvts_roundph_epi8(__m256h __A) {
  return _mm256_ipcvts_roundph_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvts_roundph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  return _mm256_mask_ipcvts_roundph_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvts_roundph_epi8(__mmask16 __A, __m256h __B) {
  return _mm256_maskz_ipcvts_roundph_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvts_roundph_epu8(__m256h __A) {
  return _mm256_ipcvts_roundph_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvts_roundph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  return _mm256_mask_ipcvts_roundph_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvts_roundph_epu8(__mmask16 __A, __m256h __B) {
  return _mm256_maskz_ipcvts_roundph_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvts_roundps_epi8(__m256 __A) {
  return _mm256_ipcvts_roundps_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvts_roundps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  return _mm256_mask_ipcvts_roundps_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvts_roundps_epi8(__mmask8 __A, __m256 __B) {
  return _mm256_maskz_ipcvts_roundps_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvts_roundps_epu8(__m256 __A) {
  return _mm256_ipcvts_roundps_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvts_roundps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  return _mm256_mask_ipcvts_roundps_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvts_roundps_epu8(__mmask8 __A, __m256 __B) {
  return _mm256_maskz_ipcvts_roundps_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvtts_roundph_epi8(__m256h __A) {
  return _mm256_ipcvtts_roundph_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvtts_roundph_epi8(__m256i __S, __mmask16 __A, __m256h __B) {
  return _mm256_mask_ipcvtts_roundph_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvtts_roundph_epi8(__mmask16 __A, __m256h __B) {
  return _mm256_maskz_ipcvtts_roundph_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvtts_roundph_epu8(__m256h __A) {
  return _mm256_ipcvtts_roundph_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvtts_roundph_epu8(__m256i __S, __mmask16 __A, __m256h __B) {
  return _mm256_mask_ipcvtts_roundph_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvtts_roundph_epu8(__mmask16 __A, __m256h __B) {
  return _mm256_maskz_ipcvtts_roundph_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvtts_roundps_epi8(__m256 __A) {
  return _mm256_ipcvtts_roundps_epi8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvtts_roundps_epi8(__m256i __S, __mmask8 __A, __m256 __B) {
  return _mm256_mask_ipcvtts_roundps_epi8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvtts_roundps_epi8(__mmask8 __A, __m256 __B) {
  return _mm256_maskz_ipcvtts_roundps_epi8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_ipcvtts_roundps_epu8(__m256 __A) {
  return _mm256_ipcvtts_roundps_epu8(__A, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_mask_ipcvtts_roundps_epu8(__m256i __S, __mmask8 __A, __m256 __B) {
  return _mm256_mask_ipcvtts_roundps_epu8(__S, __A, __B, 22); // expected-error {{invalid rounding argument}}
}

__m256i test_mm256_maskz_ipcvtts_roundps_epu8(__mmask8 __A, __m256 __B) {
  return _mm256_maskz_ipcvtts_roundps_epu8(__A, __B, 22); // expected-error {{invalid rounding argument}}
}
