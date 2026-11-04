// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +pclmul -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +pclmul -emit-llvm -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +pclmul -emit-llvm -o - -std=c++11 -fexperimental-new-constant-interpreter | FileCheck %s

#include <wmmintrin.h>
#include "builtin_test_helpers.h"

__m128i test_mm_clmulepi64_si128(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.pclmulqdq
  return _mm_clmulepi64_si128(a, b, 0);
}

// Test constexpr evaluation for _mm_clmulepi64_si128
// imm8=0x00: lower 64 bits of both operands
// Test case: 0x1 * 0x3 = 0x3 (carry-less multiplication)
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){0x1ULL, 0x0ULL}), ((__m128i){0x3ULL, 0x0ULL}), 0x00), 0x3ULL, 0x0ULL));

// imm8=0x01: upper 64 bits of first operand, lower 64 bits of second
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){0x0ULL, 0x1ULL}), ((__m128i){0x3ULL, 0x0ULL}), 0x01), 0x3ULL, 0x0ULL));

// imm8=0x10: lower 64 bits of first operand, upper 64 bits of second
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){0x1ULL, 0x0ULL}), ((__m128i){0x0ULL, 0x3ULL}), 0x10), 0x3ULL, 0x0ULL));

// imm8=0x11: upper 64 bits of both operands
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){0x0ULL, 0x1ULL}), ((__m128i){0x0ULL, 0x3ULL}), 0x11), 0x3ULL, 0x0ULL));

// Test cases with non-zero upper 64-bit results
// imm8=0x00: lower 64 bits of both operands
// 0x8000000000000000 * 0x2 = result with upper bits set
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){(long long)0x8000000000000000ULL, 0x0ULL}), ((__m128i){0x2ULL, 0x0ULL}), 0x00), 0x0ULL, 0x1ULL));

// imm8=0x01: upper 64 bits of first operand, lower 64 bits of second
// 0xFFFFFFFFFFFFFFFF * 0x2 = result with upper bits set
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){0x0ULL, (long long)0xFFFFFFFFFFFFFFFFULL}), ((__m128i){0x2ULL, 0x0ULL}), 0x01), 0xFFFFFFFFFFFFFFFEULL, 0x1ULL));

// imm8=0x10: lower 64 bits of first operand, upper 64 bits of second
// 0x1000000000000000 * 0x10 = result with upper bits set
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){(long long)0x1000000000000000ULL, 0x0ULL}), ((__m128i){0x0ULL, 0x10ULL}), 0x10), 0x0ULL, 0x1ULL));

// imm8=0x11: upper 64 bits of both operands
// 0x8000000000000001 * 0x8000000000000001 = result with upper bits set
TEST_CONSTEXPR(match_m128i(_mm_clmulepi64_si128(((__m128i){0x0ULL, (long long)0x8000000000000001ULL}), ((__m128i){0x0ULL, (long long)0x8000000000000001ULL}), 0x11), 0x1ULL, 0x4000000000000000ULL));
