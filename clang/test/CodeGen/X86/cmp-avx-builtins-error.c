// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +avx -emit-llvm -fsyntax-only -verify
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown \
// RUN: -target-feature +avx -emit-llvm -fsyntax-only -verify

#include <immintrin.h>

__m128d test_mm_cmp_pd(__m128d a, __m128d b) {
  return _mm_cmp_pd(a, b, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}

__m128d test_mm_cmp_sd(__m128d a, __m128d b) {
  return _mm_cmp_sd(a, b, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}

__m128 test_mm_cmp_ps(__m128 a, __m128 b) {
  return _mm_cmp_pd(a, b, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}

__m128 test_mm_cmp_ss(__m128 a, __m128 b) {
  return _mm_cmp_sd(a, b, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}
