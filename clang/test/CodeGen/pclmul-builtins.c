// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +pclmul -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <wmmintrin.h>

__m128i test_mm_clmulepi64_si128(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.pclmulqdq
  return _mm_clmulepi64_si128(a, b, 0);
}
