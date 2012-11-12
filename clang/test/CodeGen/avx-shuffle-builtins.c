// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

//
// Test LLVM IR codegen of shuffle instructions
//

__m256 x(__m256 a, __m256 b) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  return _mm256_shuffle_ps(a, b, 203);
}

__m128d test_mm_permute_pd(__m128d a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 1, i32 0>
  return _mm_permute_pd(a, 1);
}

__m256d test_mm256_permute_pd(__m256d a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 1, i32 0, i32 3, i32 2>
  return _mm256_permute_pd(a, 5);
}

__m128 test_mm_permute_ps(__m128 a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 1, i32 0>
  return _mm_permute_ps(a, 0x1b);
}

// Test case for PR12401
__m128 test_mm_permute_ps2(__m128 a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 2, i32 1, i32 2, i32 3>
  return _mm_permute_ps(a, 0xe6);
}

__m256 test_mm256_permute_ps(__m256 a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  return _mm256_permute_ps(a, 0x1b);
}

__m256d test_mm256_permute2f128_pd(__m256d a, __m256d b) {
  // Check if the mask is correct
  // CHECK: @llvm.x86.avx.vperm2f128.pd.256
  return _mm256_permute2f128_pd(a, b, 0x31);
}

__m256 test_mm256_permute2f128_ps(__m256 a, __m256 b) {
  // Check if the mask is correct
  // CHECK: @llvm.x86.avx.vperm2f128.ps.256
  return _mm256_permute2f128_ps(a, b, 0x13);
}

__m256i test_mm256_permute2f128_si256(__m256i a, __m256i b) {
  // Check if the mask is correct
  // CHECK: @llvm.x86.avx.vperm2f128.si.256
  return _mm256_permute2f128_si256(a, b, 0x20);
}
