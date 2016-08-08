// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse42-intrinsics-fast-isel.ll

int test_mm_cmpestra(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestra
  // CHECK: call i32 @llvm.x86.sse42.pcmpestria128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestra(A, LA, B, LB, 7);
}

int test_mm_cmpestrc(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrc
  // CHECK: call i32 @llvm.x86.sse42.pcmpestric128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrc(A, LA, B, LB, 7);
}

int test_mm_cmpestri(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestri
  // CHECK: call i32 @llvm.x86.sse42.pcmpestri128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestri(A, LA, B, LB, 7);
}

__m128i test_mm_cmpestrm(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrm
  // CHECK: call <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrm(A, LA, B, LB, 7);
}

int test_mm_cmpestro(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestro
  // CHECK: call i32 @llvm.x86.sse42.pcmpestrio128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestro(A, LA, B, LB, 7);
}

int test_mm_cmpestrs(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrs
  // CHECK: call i32 @llvm.x86.sse42.pcmpestris128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrs(A, LA, B, LB, 7);
}

int test_mm_cmpestrz(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrz
  // CHECK: call i32 @llvm.x86.sse42.pcmpestriz128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrz(A, LA, B, LB, 7);
}

__m128i test_mm_cmpgt_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi64
  // CHECK: icmp sgt <2 x i64>
  return _mm_cmpgt_epi64(A, B);
}

int test_mm_cmpistra(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistra
  // CHECK: call i32 @llvm.x86.sse42.pcmpistria128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistra(A, B, 7);
}

int test_mm_cmpistrc(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrc
  // CHECK: call i32 @llvm.x86.sse42.pcmpistric128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrc(A, B, 7);
}

int test_mm_cmpistri(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistri
  // CHECK: call i32 @llvm.x86.sse42.pcmpistri128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistri(A, B, 7);
}

__m128i test_mm_cmpistrm(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrm
  // CHECK: call <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrm(A, B, 7);
}

int test_mm_cmpistro(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistro
  // CHECK: call i32 @llvm.x86.sse42.pcmpistrio128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistro(A, B, 7);
}

int test_mm_cmpistrs(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrs
  // CHECK: call i32 @llvm.x86.sse42.pcmpistris128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrs(A, B, 7);
}

int test_mm_cmpistrz(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrz
  // CHECK: call i32 @llvm.x86.sse42.pcmpistriz128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrz(A, B, 7);
}

unsigned int test_mm_crc32_u8(unsigned int CRC, unsigned char V) {
  // CHECK-LABEL: test_mm_crc32_u8
  // CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
  return _mm_crc32_u8(CRC, V);
}

unsigned int test_mm_crc32_u16(unsigned int CRC, unsigned short V) {
  // CHECK-LABEL: test_mm_crc32_u16
  // CHECK: call i32 @llvm.x86.sse42.crc32.32.16(i32 %{{.*}}, i16 %{{.*}})
  return _mm_crc32_u16(CRC, V);
}

unsigned int test_mm_crc32_u32(unsigned int CRC, unsigned int V) {
  // CHECK-LABEL: test_mm_crc32_u32
  // CHECK: call i32 @llvm.x86.sse42.crc32.32.32(i32 %{{.*}}, i32 %{{.*}})
  return _mm_crc32_u32(CRC, V);
}

unsigned long long test_mm_crc32_u64(unsigned long long CRC, unsigned long long V) {
  // CHECK-LABEL: test_mm_crc32_u64
  // CHECK: call i64 @llvm.x86.sse42.crc32.64.64(i64 %{{.*}}, i64 %{{.*}})
  return _mm_crc32_u64(CRC, V);
}
