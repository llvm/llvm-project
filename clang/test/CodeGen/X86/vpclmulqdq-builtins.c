// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -emit-llvm -o - | FileCheck %s --check-prefix AVX
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes AVX,AVX512
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -emit-llvm -o - -std=c++11 | FileCheck %s --check-prefix AVX
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -target-feature +avx512f -emit-llvm -o - -std=c++11 | FileCheck %s --check-prefixes AVX,AVX512
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -emit-llvm -o - -std=c++11 -fexperimental-new-constant-interpreter | FileCheck %s --check-prefix AVX
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -target-feature +avx512f -emit-llvm -o - -std=c++11 -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes AVX,AVX512

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m256i test_mm256_clmulepi64_epi128(__m256i A, __m256i B) {
  // AVX: @llvm.x86.pclmulqdq.256
  return _mm256_clmulepi64_epi128(A, B, 0);
}

// Test constexpr evaluation for _mm256_clmulepi64_epi128
// Each 128-bit lane is processed independently
TEST_CONSTEXPR(match_m256i(_mm256_clmulepi64_epi128(((__m256i){0x1ULL, 0x0ULL, 0x2ULL, 0x0ULL}), ((__m256i){0x3ULL, 0x0ULL, 0x5ULL, 0x0ULL}), 0x00), 0x3ULL, 0x0ULL, 0xaULL, 0x0ULL));

#ifdef __AVX512F__
__m512i test_mm512_clmulepi64_epi128(__m512i A, __m512i B) {
  // AVX512: @llvm.x86.pclmulqdq.512
  return _mm512_clmulepi64_epi128(A, B, 0);
}

// Test constexpr evaluation for _mm512_clmulepi64_epi128
// Each 128-bit lane is processed independently
TEST_CONSTEXPR(match_m512i(_mm512_clmulepi64_epi128(((__m512i){0x1ULL, 0x0ULL, 0x2ULL, 0x0ULL, 0x4ULL, 0x0ULL, 0x8ULL, 0x0ULL}), ((__m512i){0x3ULL, 0x0ULL, 0x5ULL, 0x0ULL, 0x7ULL, 0x0ULL, 0x9ULL, 0x0ULL}), 0x00), 0x3ULL, 0x0ULL, 0xaULL, 0x0ULL, 0x1cULL, 0x0ULL, 0x48ULL, 0x0ULL));
#endif

