// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -emit-llvm -o - | FileCheck %s --check-prefix AVX
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vpclmulqdq -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes AVX,AVX512

#include <immintrin.h>

__m256i test_mm256_clmulepi64_epi128(__m256i A, __m256i B) {
  // AVX: @llvm.x86.pclmulqdq.256
  return _mm256_clmulepi64_epi128(A, B, 0);
}

#ifdef __AVX512F__
__m512i test_mm512_clmulepi64_epi128(__m512i A, __m512i B) {
  // AVX512: @llvm.x86.pclmulqdq.512
  return _mm512_clmulepi64_epi128(A, B, 0);
}
#endif

