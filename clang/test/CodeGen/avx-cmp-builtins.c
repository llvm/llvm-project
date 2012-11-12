// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

//
// Test LLVM IR codegen of cmpXY instructions
//

__m128d test_cmp_pd(__m128d a, __m128d b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.sse2.cmp.pd({{.*}}, i8 13)
  return _mm_cmp_pd(a, b, _CMP_GE_OS);
}

__m128d test_cmp_ps(__m128 a, __m128 b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.sse.cmp.ps({{.*}}, i8 13)
  return _mm_cmp_ps(a, b, _CMP_GE_OS);
}

__m256d test_cmp_pd256(__m256d a, __m256d b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.avx.cmp.pd.256({{.*}}, i8 13)
  return _mm256_cmp_pd(a, b, _CMP_GE_OS);
}

__m256d test_cmp_ps256(__m256 a, __m256 b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.avx.cmp.ps.256({{.*}}, i8 13)
  return _mm256_cmp_ps(a, b, _CMP_GE_OS);
}

__m128d test_cmp_sd(__m128d a, __m128d b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.sse2.cmp.sd({{.*}}, i8 13)
  return _mm_cmp_sd(a, b, _CMP_GE_OS);
}

__m128d test_cmp_ss(__m128 a, __m128 b) {
  // Expects that the third argument in LLVM IR is immediate expression
  // CHECK: @llvm.x86.sse.cmp.ss({{.*}}, i8 13)
  return _mm_cmp_ss(a, b, _CMP_GE_OS);
}
