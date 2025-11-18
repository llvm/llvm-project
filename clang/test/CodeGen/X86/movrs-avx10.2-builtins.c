// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-- -target-feature +movrs -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_loadrs_epi8(const __m128i * __A) {
  // CHECK-LABEL: @test_mm_loadrs_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vmovrsb128(
  return _mm_loadrs_epi8(__A);
}

__m128i test_mm_mask_loadrs_epi8(__m128i __A, __mmask16 __B, const __m128i * __C) {
  // CHECK-LABEL: @test_mm_mask_loadrs_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vmovrsb128(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_loadrs_epi8(__A, __B, __C);
}

__m128i test_mm_maskz_loadrs_epi8(__mmask16 __A, const __m128i * __B) {
  // CHECK-LABEL: @test_mm_maskz_loadrs_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vmovrsb128(
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_loadrs_epi8(__A, __B);
}

__m256i test_mm256_loadrs_epi8(const __m256i * __A) {
  // CHECK-LABEL: @test_mm256_loadrs_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vmovrsb256(
  return _mm256_loadrs_epi8(__A);
}

__m256i test_mm256_mask_loadrs_epi8(__m256i __A, __mmask32 __B, const __m256i * __C) {
  // CHECK-LABEL: @test_mm256_mask_loadrs_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vmovrsb256(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_loadrs_epi8(__A, __B, __C);
}

__m256i test_mm256_maskz_loadrs_epi8(__mmask32 __A, const __m256i * __B) {
  // CHECK-LABEL: @test_mm256_maskz_loadrs_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vmovrsb256(
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_loadrs_epi8(__A, __B);
}

__m128i test_mm_loadrs_epi32(const __m128i * __A) {
  // CHECK-LABEL: @test_mm_loadrs_epi32(
  // CHECK: call <4 x i32> @llvm.x86.avx10.vmovrsd128(
  return _mm_loadrs_epi32(__A);
}

__m128i test_mm_mask_loadrs_epi32(__m128i __A, __mmask8 __B, const __m128i * __C) {
  // CHECK-LABEL: @test_mm_mask_loadrs_epi32(
  // CHECK: call <4 x i32> @llvm.x86.avx10.vmovrsd128(
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_loadrs_epi32(__A, __B, __C);
}

__m128i test_mm_maskz_loadrs_epi32(__mmask8 __A, const __m128i * __B) {
  // CHECK-LABEL: @test_mm_maskz_loadrs_epi32(
  // CHECK: call <4 x i32> @llvm.x86.avx10.vmovrsd128(
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_loadrs_epi32(__A, __B);
}

__m256i test_mm256_loadrs_epi32(const __m256i * __A) {
  // CHECK-LABEL: @test_mm256_loadrs_epi32(
  // CHECK: call <8 x i32> @llvm.x86.avx10.vmovrsd256(
  return _mm256_loadrs_epi32(__A);
}

__m256i test_mm256_mask_loadrs_epi32(__m256i __A, __mmask8 __B, const __m256i * __C) {
  // CHECK-LABEL: @test_mm256_mask_loadrs_epi32(
  // CHECK: call <8 x i32> @llvm.x86.avx10.vmovrsd256(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_loadrs_epi32(__A, __B, __C);
}

__m256i test_mm256_maskz_loadrs_epi32(__mmask8 __A, const __m256i * __B) {
  // CHECK-LABEL: @test_mm256_maskz_loadrs_epi32(
  // CHECK: call <8 x i32> @llvm.x86.avx10.vmovrsd256(
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_loadrs_epi32(__A, __B);
}

__m128i test_mm_loadrs_epi64(const __m128i * __A) {
  // CHECK-LABEL: @test_mm_loadrs_epi64(
  // CHECK: call <2 x i64> @llvm.x86.avx10.vmovrsq128(
  return _mm_loadrs_epi64(__A);
}

__m128i test_mm_mask_loadrs_epi64(__m128i __A, __mmask8 __B, const __m128i * __C) {
  // CHECK-LABEL: @test_mm_mask_loadrs_epi64(
  // CHECK: call <2 x i64> @llvm.x86.avx10.vmovrsq128(
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_loadrs_epi64(__A, __B, __C);
}

__m128i test_mm_maskz_loadrs_epi64(__mmask8 __A, const __m128i * __B) {
  // CHECK-LABEL: @test_mm_maskz_loadrs_epi64(
  // CHECK: call <2 x i64> @llvm.x86.avx10.vmovrsq128(
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_loadrs_epi64(__A, __B);
}

__m256i test_mm256_loadrs_epi64(const __m256i * __A) {
  // CHECK-LABEL: @test_mm256_loadrs_epi64(
  // CHECK: call <4 x i64> @llvm.x86.avx10.vmovrsq256(
  return _mm256_loadrs_epi64(__A);
}

__m256i test_mm256_mask_loadrs_epi64(__m256i __A, __mmask8 __B, const __m256i * __C) {
  // CHECK-LABEL: @test_mm256_mask_loadrs_epi64(
  // CHECK: call <4 x i64> @llvm.x86.avx10.vmovrsq256(
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_loadrs_epi64(__A, __B, __C);
}

__m256i test_mm256_maskz_loadrs_epi64(__mmask8 __A, const __m256i * __B) {
  // CHECK-LABEL: @test_mm256_maskz_loadrs_epi64(
  // CHECK: call <4 x i64> @llvm.x86.avx10.vmovrsq256(
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_loadrs_epi64(__A, __B);
}

__m128i test_mm_loadrs_epi16(const __m128i * __A) {
  // CHECK-LABEL: @test_mm_loadrs_epi16(
  // CHECK: call <8 x i16> @llvm.x86.avx10.vmovrsw128(
  return _mm_loadrs_epi16(__A);
}

__m128i test_mm_mask_loadrs_epi16(__m128i __A, __mmask8 __B, const __m128i * __C) {
  // CHECK-LABEL: @test_mm_mask_loadrs_epi16(
  // CHECK: call <8 x i16> @llvm.x86.avx10.vmovrsw128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_loadrs_epi16(__A, __B, __C);
}

__m128i test_mm_maskz_loadrs_epi16(__mmask8 __A, const __m128i * __B) {
  // CHECK-LABEL: @test_mm_maskz_loadrs_epi16(
  // CHECK: call <8 x i16> @llvm.x86.avx10.vmovrsw128(
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_loadrs_epi16(__A, __B);
}

__m256i test_mm256_loadrs_epi16(const __m256i * __A) {
  // CHECK-LABEL: @test_mm256_loadrs_epi16(
  // CHECK: call <16 x i16> @llvm.x86.avx10.vmovrsw256(
  return _mm256_loadrs_epi16(__A);
}

__m256i test_mm256_mask_loadrs_epi16(__m256i __A, __mmask16 __B, const __m256i * __C) {
  // CHECK-LABEL: @test_mm256_mask_loadrs_epi16(
  // CHECK: call <16 x i16> @llvm.x86.avx10.vmovrsw256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_loadrs_epi16(__A, __B, __C);
}

__m256i test_mm256_maskz_loadrs_epi16(__mmask16 __A, const __m256i * __B) {
  // CHECK-LABEL: @test_mm256_maskz_loadrs_epi16(
  // CHECK: call <16 x i16> @llvm.x86.avx10.vmovrsw256(
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_loadrs_epi16(__A, __B);
}
