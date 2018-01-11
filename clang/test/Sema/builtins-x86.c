// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsyntax-only -verify %s

typedef long long __m128i __attribute__((__vector_size__(16)));
typedef float __m128 __attribute__((__vector_size__(16)));
typedef double __m128d __attribute__((__vector_size__(16)));

typedef long long __m256i __attribute__((__vector_size__(32)));
typedef float __m256 __attribute__((__vector_size__(32)));
typedef double __m256d __attribute__((__vector_size__(32)));

typedef long long __m512i __attribute__((__vector_size__(64)));
typedef float __m512 __attribute__((__vector_size__(64)));
typedef double __m512d __attribute__((__vector_size__(64)));

typedef unsigned char __mmask8;
typedef unsigned short __mmask16;
typedef unsigned int __mmask32;

__m128 test__builtin_ia32_cmpps(__m128 __a, __m128 __b) {
  __builtin_ia32_cmpps(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128d test__builtin_ia32_cmppd(__m128d __a, __m128d __b) {
  __builtin_ia32_cmppd(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128 test__builtin_ia32_cmpss(__m128 __a, __m128 __b) {
  __builtin_ia32_cmpss(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__m128d test__builtin_ia32_cmpsd(__m128d __a, __m128d __b) {
  __builtin_ia32_cmpsd(__a, __b, 32); // expected-error {{argument should be a value from 0 to 31}}
}

__mmask16 test__builtin_ia32_cmpps512_mask(__m512d __a, __m512d __b) {
  __builtin_ia32_cmpps512_mask(__a, __b, 32, -1, 4); // expected-error {{argument should be a value from 0 to 31}}
}

__mmask8 test__builtin_ia32_cmppd512_mask(__m512d __a, __m512d __b) {
  __builtin_ia32_cmppd512_mask(__a, __b, 32, -1, 4); // expected-error {{argument should be a value from 0 to 31}}
}

__m128i test__builtin_ia32_vpcomub(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomub(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomuw(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuw(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomud(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomud(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomuq(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuq(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomb(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomub(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomw(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuw(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomd(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomud(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__m128i test__builtin_ia32_vpcomq(__m128i __a, __m128i __b) {
  __builtin_ia32_vpcomuq(__a, __b, 8); // expected-error {{argument should be a value from 0 to 7}}
}

__mmask16 test__builtin_ia32_cmpps512_mask_rounding(__m512 __a, __m512 __b, __mmask16 __u) {
  __builtin_ia32_cmpps512_mask(__a, __b, 0, __u, 0); // expected-error {{invalid rounding argument}}
}

__m128i test_mm_mask_i32gather_epi32(__m128i a, int const *b, __m128i c, __m128i mask) {
  return __builtin_ia32_gatherd_d(a, b, c, mask, 5); // expected-error {{scale argument must be 1, 2, 4, or 8}}
}

__m512i _mm512_mask_prefetch_i32gather_ps(__m512i index, __mmask16 mask, int const *addr) {
  return __builtin_ia32_gatherpfdps(mask, index, addr, 5, 1); // expected-error {{scale argument must be 1, 2, 4, or 8}}
}

__m512 _mm512_mask_prefetch_i32gather_ps_2(__m512i index, __mmask16 mask, int const *addr) {
  return __builtin_ia32_gatherpfdps(mask, index, addr, 1, 1); // expected-error {{argument should be a value from 2 to 3}}
}

__m512i test_mm512_mask_shldi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  return __builtin_ia32_vpshldq512_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m512i test_mm512_mask_shldi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  return __builtin_ia32_vpshldd512_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m512i test_mm512_mask_shldi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  return __builtin_ia32_vpshldw512_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m512i test_mm512_mask_shrdi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  return __builtin_ia32_vpshrdq512_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m512i test_mm512_mask_shrdi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  return __builtin_ia32_vpshrdd512_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m512i test_mm512_mask_shrdi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  return __builtin_ia32_vpshrdw512_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m256i test_mm256_mask_shldi_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  return __builtin_ia32_vpshldq256_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m128i test_mm128_mask_shldi_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  return __builtin_ia32_vpshldq128_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m256i test_mm256_mask_shldi_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  return __builtin_ia32_vpshldd256_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m128i test_mm128_mask_shldi_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  return __builtin_ia32_vpshldd128_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m256i test_mm256_mask_shldi_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  return __builtin_ia32_vpshldw256_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m128i test_mm128_mask_shldi_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  return __builtin_ia32_vpshldw128_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m256i test_mm256_mask_shrdi_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  return __builtin_ia32_vpshrdq256_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m128i test_mm128_mask_shrdi_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  return __builtin_ia32_vpshrdq128_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m256i test_mm256_mask_shrdi_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  return __builtin_ia32_vpshrdd256_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m128i test_mm128_mask_shrdi_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  return __builtin_ia32_vpshrdd128_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m256i test_mm256_mask_shrdi_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  return __builtin_ia32_vpshrdw256_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}

__m128i test_mm128_mask_shrdi_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  return __builtin_ia32_vpshrdw128_mask(__A, __B, 1024, __S, __U); // expected-error {{argument should be a value from 0 to 255}}
}
