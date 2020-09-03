// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -ffreestanding -Wcast-qual %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ffreestanding -Wcast-qual %s -verify
// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -ffreestanding -flax-vector-conversions=none -Wcast-qual %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ffreestanding -flax-vector-conversions=none -Wcast-qual %s -verify
// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only -ffreestanding -Wcast-qual -x c++ %s -verify
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ffreestanding -Wcast-qual -x c++ %s -verify
// expected-no-diagnostics

// Include the metaheader that includes all x86 intrinsic headers.
#include <x86intrin.h>

void __attribute__((__target__("mmx"))) mm_empty_wrap(void) {
  _mm_empty();
}

__m128 __attribute__((__target__("sse"))) mm_add_ss_wrap(__m128 a, __m128 b) {
  return _mm_add_ss(a, b);
}

void __attribute__((__target__("sse"))) mm_prefetch_wrap(const void *p) {
  _mm_prefetch(p, 0x3);
}

__m128d __attribute__((__target__("sse2"))) mm_sqrt_sd_wrap(__m128d a, __m128d b) {
  return _mm_sqrt_sd(a, b);
}

void __attribute__((__target__("sse3"))) mm_mwait_wrap(int a) {
  _mm_mwait(0, 0);
}

__m64 __attribute__((__target__("ssse3"))) mm_abs_pi8_wrap(__m64 a) {
  return _mm_abs_pi8(a);
}

__m128i __attribute__((__target__("sse4.1"))) mm_minpos_epu16_wrap(__m128i v) {
  return _mm_minpos_epu16(v);
}

unsigned int __attribute__((__target__("sse4.2"))) mm_crc32_u8_wrap(unsigned int c, unsigned char d) {
  return _mm_crc32_u8(c, d);
}

__m128i __attribute__((__target__("aes"))) mm_aesenc_si128_wrap(__m128i v, __m128i r) {
  return _mm_aesenc_si128(v, r);
}

__m256d __attribute__((__target__("avx"))) mm256_add_pd_wrap(__m256d a, __m256d b) {
  return _mm256_add_pd(a, b);
}

__m256i __attribute__((__target__("avx2"))) mm256_abs_epi8_wrap(__m256i a) {
  return _mm256_abs_epi8(a);
}

unsigned short __attribute__((__target__("bmi"))) tzcnt_u16_wrap(unsigned short x) {
  return __tzcnt_u16(x);
}

unsigned int __attribute__((__target__("bmi2"))) bzhi_u32_wrap(unsigned int x, unsigned int y) {
  return _bzhi_u32(x, y);
}

unsigned short __attribute__((__target__("lzcnt"))) lzcnt16_wrap(unsigned short x) {
  return __lzcnt16(x);
}

__m256d __attribute__((__target__("fma"))) mm256_fmsubadd_pd_wrap(__m256d a, __m256d b, __m256d c) {
  return _mm256_fmsubadd_pd(a, b, c);
}

__m512i __attribute__((__target__("avx512f"))) mm512_setzero_si512_wrap(void) {
  return _mm512_setzero_si512();
}

__mmask8 __attribute__((__target__("avx512vl"))) mm_cmpeq_epi32_mask_wrap(__m128i a, __m128i b) {
  return _mm_cmpeq_epi32_mask(a, b);
}

__m512i __attribute__((__target__("avx512dq"))) mm512_mullo_epi64_wrap(__m512i a, __m512i b) {
  return _mm512_mullo_epi64(a, b);
}

__mmask16 __attribute__((__target__("avx512vl,avx512bw"))) mm_cmpeq_epi8_mask_wrap(__m128i a, __m128i b) {
  return _mm_cmpeq_epi8_mask(a, b);
}

__m256i __attribute__((__target__("avx512vl,avx512dq"))) mm256_mullo_epi64_wrap(__m256i a, __m256i b) {
  return _mm256_mullo_epi64(a, b);
}

int __attribute__((__target__("rdrnd"))) rdrand16_step_wrap(unsigned short *p) {
  return _rdrand16_step(p);
}

#if defined(__x86_64__)
unsigned int __attribute__((__target__("fsgsbase"))) readfsbase_u32_wrap(void) {
  return _readfsbase_u32();
}
#endif

unsigned int __attribute__((__target__("rtm"))) xbegin_wrap(void) {
  return _xbegin();
}

__m128i __attribute__((__target__("sha"))) mm_sha1nexte_epu32_wrap(__m128i x, __m128i y) {
  return _mm_sha1nexte_epu32(x, y);
}

int __attribute__((__target__("rdseed"))) rdseed16_step_wrap(unsigned short *p) {
  return _rdseed16_step(p);
}

__m128i __attribute__((__target__("sse4a"))) mm_extract_si64_wrap(__m128i x, __m128i y) {
  return _mm_extract_si64(x, y);
}

__m128 __attribute__((__target__("fma4"))) mm_macc_ps_wrap(__m128 a, __m128 b, __m128 c) {
  return _mm_macc_ps(a, b, c);
}

__m256 __attribute__((__target__("xop"))) mm256_frcz_ps_wrap(__m256 a) {
  return _mm256_frcz_ps(a);
}

unsigned int __attribute__((__target__("tbm"))) blcfill_u32_wrap(unsigned int a) {
  return __blcfill_u32(a);
}

__m128 __attribute__((__target__("f16c"))) mm_cvtph_ps_wrap(__m128i a) {
  return _mm_cvtph_ps(a);
}

int __attribute__((__target__("rtm"))) xtest_wrap(void) {
  return _xtest();
}
