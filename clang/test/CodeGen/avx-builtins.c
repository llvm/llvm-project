// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

//
// Test LLVM IR codegen of shuffle instructions
//

__m256 test__mm256_loadu_ps(void* p) {
  // CHECK: load <8 x float>* %{{.*}}, align 1
  return _mm256_loadu_ps(p);
}

__m256d test__mm256_loadu_pd(void* p) {
  // CHECK: load <4 x double>* %{{.*}}, align 1
  return _mm256_loadu_pd(p);
}

__m256i test__mm256_loadu_si256(void* p) {
  // CHECK: load <4 x i64>* %{{.+}}, align 1
  return _mm256_loadu_si256(p);
}

__m128i test_mm_cmpestrm(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestrm128
  return _mm_cmpestrm(A, LA, B, LB, 7);
}

int test_mm_cmpestri(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestri128
  return _mm_cmpestri(A, LA, B, LB, 7);
}

int test_mm_cmpestra(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestria128
  return _mm_cmpestra(A, LA, B, LB, 7);
}

int test_mm_cmpestrc(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestric128
  return _mm_cmpestrc(A, LA, B, LB, 7);
}

int test_mm_cmpestro(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestrio128
  return _mm_cmpestro(A, LA, B, LB, 7);
}

int test_mm_cmpestrs(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestris128
  return _mm_cmpestrs(A, LA, B, LB, 7);
}

int test_mm_cmpestrz(__m128i A, int LA, __m128i B, int LB) {
  // CHECK: @llvm.x86.sse42.pcmpestriz128
  return _mm_cmpestrz(A, LA, B, LB, 7);
}

__m128i test_mm_cmpistrm(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistrm128
  return _mm_cmpistrm(A, B, 7);
}

int test_mm_cmpistri(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistri128
  return _mm_cmpistri(A, B, 7);
}

int test_mm_cmpistra(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistria128
  return _mm_cmpistra(A, B, 7);
}

int test_mm_cmpistrc(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistric128
  return _mm_cmpistrc(A, B, 7);
}

int test_mm_cmpistro(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistrio128
  return _mm_cmpistro(A, B, 7);
}

int test_mm_cmpistrs(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistris128
  return _mm_cmpistrs(A, B, 7);
}

int test_mm_cmpistrz(__m128i A, __m128i B) {
  // CHECK: @llvm.x86.sse42.pcmpistriz128
  return _mm_cmpistrz(A, B, 7);
}
