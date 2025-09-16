// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -O0 -target-cpu skylake-avx512 -std=c++17 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -O0 -target-cpu skylake-avx512 -std=c++17 -fexperimental-new-constant-interpreter -emit-llvm -o - %s | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

//
// AVX256 Insert Tests
//

__m256 test_mm256_insertf32x4(__m256 A, __m128 B) {
  // CHECK-LABEL: test_mm256_insertf32x4
  return _mm256_insertf32x4(A, B, 1);
}

// Insert 128-bit float vector into upper lane
TEST_CONSTEXPR(match_m256(_mm256_insertf32x4(_mm256_set1_ps(1.0f), _mm_set_ps(40.0f, 30.0f, 20.0f, 10.0f), 1), 1.0f, 1.0f, 1.0f, 1.0f, 10.0f, 20.0f, 30.0f, 40.0f));

__m256i test_mm256_inserti32x4(__m256i A, __m128i B) {
  // CHECK-LABEL: test_mm256_inserti32x4
  return _mm256_inserti32x4(A, B, 0);
}

// Insert 128-bit integer vector into lower lane
TEST_CONSTEXPR(match_v8si(_mm256_inserti32x4(_mm256_set1_epi32(1), _mm_set_epi32(40, 30, 20, 10), 0), 10, 20, 30, 40, 1, 1, 1, 1));

//
// AVX256 Masked Insert Test
//

__m256 test_mm256_maskz_insertf32x4(__mmask8 U, __m256 A, __m128 B) {
  // CHECK-LABEL: test_mm256_maskz_insertf32x4
  return _mm256_maskz_insertf32x4(U, A, B, 1);
}

// Test zero mask produces all zeros
TEST_CONSTEXPR(match_m256(
    _mm256_maskz_insertf32x4(0x00, _mm256_set1_ps(1.0f),
                             _mm_set_ps(40.0f, 30.0f, 20.0f, 10.0f), 1),
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));

//
// AVX Legacy Insert Test
//

__m256 test_mm256_insertf128_ps(__m256 A, __m128 B) {
  // CHECK-LABEL: test_mm256_insertf128_ps
  return _mm256_insertf128_ps(A, B, 1);
}

// Legacy insertf128 into upper lane
TEST_CONSTEXPR(match_m256(_mm256_insertf128_ps(_mm256_set1_ps(1.0f), _mm_set1_ps(7.0f), 1), 1.0f, 1.0f, 1.0f, 1.0f, 7.0f, 7.0f, 7.0f, 7.0f));

//
//AVX512 Insert Tests
//

__m512 test_mm512_insertf32x4(__m512 A, __m128 B) {
  // CHECK-LABEL: test_mm512_insertf32x4
  return _mm512_insertf32x4(A, B, 3);
}

// Insert 128-bit into highest lane of 512-bit vector
TEST_CONSTEXPR(match_m512(_mm512_insertf32x4(_mm512_set1_ps(1.0f), _mm_set1_ps(5.0f), 3), 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 5.0f, 5.0f, 5.0f, 5.0f));

__m512 test_mm512_insertf32x8(__m512 A, __m256 B) {
  // CHECK-LABEL: test_mm512_insertf32x8
  return _mm512_insertf32x8(A, B, 1);
}

// Insert 256-bit into upper half of 512-bit vector
TEST_CONSTEXPR(match_m512(_mm512_insertf32x8(_mm512_set1_ps(1.0f), _mm256_set1_ps(2.0f), 1), 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f));

//
// AVX512 Masked Insert Test
//

__m512 test_mm512_maskz_insertf32x4(__mmask16 U, __m512 A, __m128 B) {
  // CHECK-LABEL: test_mm512_maskz_insertf32x4
  return _mm512_maskz_insertf32x4(U, A, B, 3);
}

// Test zero mask produces all zeros
TEST_CONSTEXPR(match_m512(
    _mm512_maskz_insertf32x4(0x0000, _mm512_set1_ps(1.0f), _mm_set1_ps(5.0f), 3),
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
