// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10.2-512 \
// RUN: -Wno-invalid-feature-combination -verify -fsyntax-only
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10.2-512 \
// RUN: -Wno-invalid-feature-combination -verify -fsyntax-only

#include <immintrin.h>

__m128bh test_mm_minmax_pbh(__m128bh __A, __m128bh __B) {
  return _mm_minmax_pbh(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128bh test_mm_mask_minmax_pbh(__m128bh __A, __mmask8 __B, __m128bh __C, __m128bh __D) {
  return _mm_mask_minmax_pbh(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256bh test_mm256_minmax_pbh(__m256bh __A, __m256bh __B) {
  return _mm256_minmax_pbh(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256bh test_mm256_mask_minmax_pbh(__m256bh __A, __mmask16 __B, __m256bh __C, __m256bh __D) {
  return _mm256_mask_minmax_pbh(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128d test_mm_minmax_pd(__m128d __A, __m128d __B) {
  return _mm_minmax_pd(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128d test_mm_mask_minmax_pd(__m128d __A, __mmask8 __B, __m128d __C, __m128d __D) {
  return _mm_mask_minmax_pd(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256d test_mm256_minmax_pd(__m256d __A, __m256d __B) {
  return _mm256_minmax_pd(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256d test_mm256_mask_minmax_pd(__m256d __A, __mmask8 __B, __m256d __C, __m256d __D) {
  return _mm256_mask_minmax_pd(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128h test_mm_minmax_ph(__m128h __A, __m128h __B) {
  return _mm_minmax_ph(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128h test_mm_mask_minmax_ph(__m128h __A, __mmask8 __B, __m128h __C, __m128h __D) {
  return _mm_mask_minmax_ph(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256h test_mm256_minmax_ph(__m256h __A, __m256h __B) {
  return _mm256_minmax_ph(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256h test_mm256_mask_minmax_ph(__m256h __A, __mmask16 __B, __m256h __C, __m256h __D) {
  return _mm256_mask_minmax_ph(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128 test_mm_minmax_ps(__m128 __A, __m128 __B) {
  return _mm_minmax_ps(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128 test_mm_mask_minmax_ps(__m128 __A, __mmask8 __B, __m128 __C, __m128 __D) {
  return _mm_mask_minmax_ps(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256 test_mm256_minmax_ps(__m256 __A, __m256 __B) {
  return _mm256_minmax_ps(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m256 test_mm256_mask_minmax_ps(__m256 __A, __mmask8 __B, __m256 __C, __m256 __D) {
  return _mm256_mask_minmax_ps(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m512bh test_mm512_minmax_pbh(__m512bh __A, __m512bh __B) {
  return _mm512_minmax_pbh(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m512bh test_mm512_mask_minmax_pbh(__m512bh __A, __mmask32 __B, __m512bh __C, __m512bh __D) {
  return _mm512_mask_minmax_pbh(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m512d test_mm512_minmax_pd(__m512d __A, __m512d __B) {
  return _mm512_minmax_pd(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m512h test_mm512_mask_minmax_ph(__m512h __A, __mmask32 __B, __m512h __C, __m512h __D) {
  return _mm512_mask_minmax_ph(__A, __B, __C, __D, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m512 test_mm512_minmax_ps(__m512 __A, __m512 __B) {
  return _mm512_minmax_ps(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128d test_mm_minmax_sd(__m128d __A, __m128d __B) {
  return _mm_minmax_sd(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128h test_mm_minmax_sh(__m128h __A, __m128h __B) {
  return _mm_minmax_sh(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m128 test_mm_minmax_ss(__m128 __A, __m128 __B) {
  return _mm_minmax_ss(__A, __B, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

__m512d test_mm512_minmax_round_pd(__m512d __A, __m512d __B) {
  return _mm512_minmax_round_pd(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m512h test_mm512_minmax_round_ph(__m512h __A, __m512h __B) {
  return _mm512_minmax_round_ph(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m512 test_mm512_minmax_round_ps(__m512 __A, __m512 __B) {
  return _mm512_minmax_round_ps(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m256d test_mm256_minmax_round_pd(__m256d __A, __m256d __B) {
  return _mm256_minmax_round_pd(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m256h test_mm256_minmax_round_ph(__m256h __A, __m256h __B) {
  return _mm256_minmax_round_ph(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m256 test_mm256_minmax_round_ps(__m256 __A, __m256 __B) {
  return _mm256_minmax_round_ps(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}
__m128d test_mm_minmax_round_sd(__m128d __A, __m128d __B) {
  return _mm_minmax_round_sd(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m128h test_mm_minmax_round_sh(__m128h __A, __m128h __B) {
  return _mm_minmax_round_sh(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}

__m128 test_mm_minmax_round_ss(__m128 __A, __m128 __B) {
  return _mm_minmax_round_ss(__A, __B, 127, 11); // expected-error {{invalid rounding argument}}
}
