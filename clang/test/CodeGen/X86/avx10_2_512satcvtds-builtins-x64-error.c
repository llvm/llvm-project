// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +avx10.2-512 -emit-llvm -Wall -Werror -verify

#include <immintrin.h>
#include <stddef.h>

long long test_mm_cvttssd_si64(__m128d __A) {
  return _mm_cvtts_roundsd_si64(__A, 22); // expected-error {{invalid rounding argument}}
}

long long test_mm_cvttssd_i64(__m128d __A) {
  return _mm_cvtts_roundsd_i64(__A, 22); // expected-error {{invalid rounding argument}}
}

unsigned long long test_mm_cvttssd_u64(__m128d __A) {
  return _mm_cvtts_roundsd_u64(__A, 22); // expected-error {{invalid rounding argument}}
}

float test_mm_cvttsss_i64(__m128 __A) {
  return _mm_cvtts_roundss_i64(__A, 22); // expected-error {{invalid rounding argument}}
}

long long test_mm_cvttsss_si64(__m128 __A) {
  return _mm_cvtts_roundss_si64(__A, 22); // expected-error {{invalid rounding argument}}
}

unsigned long long test_mm_cvttsss_u64(__m128 __A) {
  return _mm_cvtts_roundss_u64(__A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_cvtts_roundpd_epi64(__m512d A) {
  return _mm512_cvtts_roundpd_epi64( A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_cvtts_roundpd_epi64(__m512i W, __mmask8 U, __m512d A) {
  return _mm512_mask_cvtts_roundpd_epi64( W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_cvtts_roundpd_epi64(__mmask8 U, __m512d A) {
  return _mm512_maskz_cvtts_roundpd_epi64( U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_cvtts_roundpd_epu64(__m512d A) {
  return _mm512_cvtts_roundpd_epu64( A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_cvtts_roundpd_epu64(__m512i W, __mmask8 U, __m512d A) {
  return _mm512_mask_cvtts_roundpd_epu64( W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_cvtts_roundpd_epu64(__mmask8 U, __m512d A) {
  return _mm512_maskz_cvtts_roundpd_epu64( U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_cvtts_roundps_epi64(__m256 A) {
  return _mm512_cvtts_roundps_epi64( A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_cvtts_roundps_epi64(__m512i W, __mmask8 U, __m256 A) {
  return _mm512_mask_cvtts_roundps_epi64( W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_cvtts_roundps_epi64(__mmask8 U, __m256 A) {
  return _mm512_maskz_cvtts_roundps_epi64( U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_cvtts_roundps_epu64(__m256 A) {
  return _mm512_cvtts_roundps_epu64( A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_mask_cvtts_roundps_epu64(__m512i W, __mmask8 U, __m256 A) {
  return _mm512_mask_cvtts_roundps_epu64( W, U, A, 22); // expected-error {{invalid rounding argument}}
}

__m512i test_mm512_maskz_cvtts_roundps_epu64(__mmask8 U, __m256 A) {
  return _mm512_maskz_cvtts_roundps_epu64( U, A, 22); // expected-error {{invalid rounding argument}}
}
