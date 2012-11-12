// RUN: %clang_cc1 -ffreestanding -triple i386-apple-darwin9 -target-cpu pentium4 -target-feature +sse4a -g -emit-llvm %s -o - | FileCheck %s

#include <ammintrin.h>

__m128i test_extracti_si64(__m128i x) {
  return _mm_extracti_si64(x, 3, 2);
// CHECK: @test_extracti_si64
// CHECK: @llvm.x86.sse4a.extrqi(<2 x i64> %{{[^,]+}}, i8 3, i8 2)
}

__m128i test_extract_si64(__m128i x, __m128i y) {
  return _mm_extract_si64(x, y);
// CHECK: @test_extract_si64
// CHECK: @llvm.x86.sse4a.extrq(<2 x i64> %{{[^,]+}}, <16 x i8> %{{[^,]+}})
}

__m128i test_inserti_si64(__m128i x, __m128i y) {
  return _mm_inserti_si64(x, y, 5, 6);
// CHECK: @test_inserti_si64
// CHECK: @llvm.x86.sse4a.insertqi(<2 x i64> %{{[^,]+}}, <2 x i64> %{{[^,]+}}, i8 5, i8 6)
}

__m128i test_insert_si64(__m128i x, __m128i y) {
  return _mm_insert_si64(x, y);
// CHECK: @test_insert_si64
// CHECK: @llvm.x86.sse4a.insertq(<2 x i64> %{{[^,]+}}, <2 x i64> %{{[^,]+}})
}

void test_stream_sd(double *p, __m128d a) {
  _mm_stream_sd(p, a);
// CHECK: @test_stream_sd
// CHECK: @llvm.x86.sse4a.movnt.sd(i8* %{{[^,]+}}, <2 x double> %{{[^,]+}})
}

void test_stream_ss(float *p, __m128 a) {
  _mm_stream_ss(p, a);
// CHECK: @test_stream_ss
// CHECK: @llvm.x86.sse4a.movnt.ss(i8* %{{[^,]+}}, <4 x float> %{{[^,]+}})
}
