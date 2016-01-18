// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_mm_cmpgt_epi8(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi8
  // CHECK: icmp sgt <16 x i8>
  // CHECK-ASM: pcmpgtb %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpgt_epi8(A, B);
}

__m128i test_mm_cmpgt_epi16(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi16
  // CHECK: icmp sgt <8 x i16>
  // CHECK-ASM: pcmpgtw %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpgt_epi16(A, B);
}

__m128i test_mm_cmpgt_epi32(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi32
  // CHECK: icmp sgt <4 x i32>
  // CHECK-ASM: pcmpgtd %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpgt_epi32(A, B);
}

__m128i test_mm_cmpgt_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi64
  // CHECK: icmp sgt <2 x i64>
  // CHECK-ASM: pcmpgtq %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpgt_epi64(A, B);
}

int test_mm_cmpestra(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestra
  // CHECK: @llvm.x86.sse42.pcmpestria128
  // CHECK-ASM: pcmpestri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestra(A, LA, B, LB, 7);
}

int test_mm_cmpestrc(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrc
  // CHECK: @llvm.x86.sse42.pcmpestric128
  // CHECK-ASM: pcmpestri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestrc(A, LA, B, LB, 7);
}

int test_mm_cmpestri(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestri
  // CHECK: @llvm.x86.sse42.pcmpestri128
  // CHECK-ASM: pcmpestri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestri(A, LA, B, LB, 7);
}

__m128i test_mm_cmpestrm(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrm
  // CHECK: @llvm.x86.sse42.pcmpestrm128
  // CHECK-ASM: pcmpestrm $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestrm(A, LA, B, LB, 7);
}

int test_mm_cmpestro(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestro
  // CHECK: @llvm.x86.sse42.pcmpestrio128
  // CHECK-ASM: pcmpestri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestro(A, LA, B, LB, 7);
}

int test_mm_cmpestrs(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrs
  // CHECK: @llvm.x86.sse42.pcmpestris128
  // CHECK-ASM: pcmpestri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestrs(A, LA, B, LB, 7);
}

int test_mm_cmpestrz(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrz
  // CHECK: @llvm.x86.sse42.pcmpestriz128
  // CHECK-ASM: pcmpestri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpestrz(A, LA, B, LB, 7);
}

int test_mm_cmpistra(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistra
  // CHECK: @llvm.x86.sse42.pcmpistria128
  // CHECK-ASM: pcmpistri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistra(A, B, 7);
}

int test_mm_cmpistrc(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrc
  // CHECK: @llvm.x86.sse42.pcmpistric128
  // CHECK-ASM: pcmpistri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistrc(A, B, 7);
}

int test_mm_cmpistri(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistri
  // CHECK: @llvm.x86.sse42.pcmpistri128
  // CHECK-ASM: pcmpistri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistri(A, B, 7);
}

__m128i test_mm_cmpistrm(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrm
  // CHECK: @llvm.x86.sse42.pcmpistrm128
  // CHECK-ASM: pcmpistrm $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistrm(A, B, 7);
}

int test_mm_cmpistro(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistro
  // CHECK: @llvm.x86.sse42.pcmpistrio128
  // CHECK-ASM: pcmpistri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistro(A, B, 7);
}

int test_mm_cmpistrs(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrs
  // CHECK: @llvm.x86.sse42.pcmpistris128
  // CHECK-ASM: pcmpistri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistrs(A, B, 7);
}

int test_mm_cmpistrz(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrz
  // CHECK: @llvm.x86.sse42.pcmpistriz128
  // CHECK-ASM: pcmpistri $7, %xmm{{.*}}, %xmm{{.*}}
  return _mm_cmpistrz(A, B, 7);
}

unsigned int test_mm_crc32_u8(unsigned int CRC, unsigned char V) {
  // CHECK-LABEL: test_mm_crc32_u8
  // CHECK: call i32 @llvm.x86.sse42.crc32.32.8
  // CHECK-ASM: crc32
  return _mm_crc32_u8(CRC, V);
}

unsigned int test_mm_crc32_u16(unsigned int CRC, unsigned short V) {
  // CHECK-LABEL: test_mm_crc32_u16
  // CHECK: call i32 @llvm.x86.sse42.crc32.32.16
  // CHECK-ASM: crc32
  return _mm_crc32_u16(CRC, V);
}

unsigned int test_mm_crc32_u32(unsigned int CRC, unsigned int V) {
  // CHECK-LABEL: test_mm_crc32_u32
  // CHECK: call i32 @llvm.x86.sse42.crc32.32.32
  // CHECK-ASM: crc32
  return _mm_crc32_u32(CRC, V);
}

unsigned int test_mm_crc32_u64(unsigned long long CRC, unsigned long long V) {
  // CHECK-LABEL: test_mm_crc32_u64
  // CHECK: call i64 @llvm.x86.sse42.crc32.64.64
  // CHECK-ASM: crc32
  return _mm_crc32_u64(CRC, V);
}
