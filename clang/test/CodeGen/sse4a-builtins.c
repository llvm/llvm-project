// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4a -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_extracti_si64(__m128i x) {
  // CHECK-LABEL: test_extracti_si64
  // CHECK: call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %{{[^,]+}}, i8 3, i8 2)
  return _mm_extracti_si64(x, 3, 2);
}

__m128i test_extract_si64(__m128i x, __m128i y) {
  // CHECK-LABEL: test_extract_si64
  // CHECK: call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %{{[^,]+}}, <16 x i8> %{{[^,]+}})
  return _mm_extract_si64(x, y);
}

__m128i test_inserti_si64(__m128i x, __m128i y) {
  // CHECK-LABEL: test_inserti_si64
  // CHECK: call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %{{[^,]+}}, <2 x i64> %{{[^,]+}}, i8 5, i8 6)
  return _mm_inserti_si64(x, y, 5, 6);
}

__m128i test_insert_si64(__m128i x, __m128i y) {
  // CHECK-LABEL: test_insert_si64
  // CHECK: call <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64> %{{[^,]+}}, <2 x i64> %{{[^,]+}})
  return _mm_insert_si64(x, y);
}

void test_stream_sd(double *p, __m128d a) {
  // CHECK-LABEL: test_stream_sd
  // CHECK: call void @llvm.x86.sse4a.movnt.sd(i8* %{{[^,]+}}, <2 x double> %{{[^,]+}})
  _mm_stream_sd(p, a);
}

void test_stream_ss(float *p, __m128 a) {
  // CHECK-LABEL: test_stream_ss
  // CHECK: call void @llvm.x86.sse4a.movnt.ss(i8* %{{[^,]+}}, <4 x float> %{{[^,]+}})
  _mm_stream_ss(p, a);
}
