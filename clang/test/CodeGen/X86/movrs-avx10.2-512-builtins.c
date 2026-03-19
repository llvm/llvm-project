// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-- -target-feature +movrs -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_loadrs_epi8(const __m512i * __A) {
  // CHECK-LABEL: @test_mm512_loadrs_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vmovrsb512(
  return _mm512_loadrs_epi8(__A);
}

__m512i test_mm512_mask_loadrs_epi8(__m512i __A, __mmask64 __B, const __m512i * __C) {
  // CHECK-LABEL: @test_mm512_mask_loadrs_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vmovrsb512(
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_mask_loadrs_epi8(__A, __B, __C);
}

__m512i test_mm512_maskz_loadrs_epi8(__mmask64 __A, const __m512i * __B) {
  // CHECK-LABEL: @test_mm512_maskz_loadrs_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vmovrsb512(
  // CHECK: store <8 x i64> zeroinitializer
  // CHECK: select <64 x i1> %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}
  return _mm512_maskz_loadrs_epi8(__A, __B);
}

__m512i test_mm512_loadrs_epi32(const __m512i * __A) {
  // CHECK-LABEL: @test_mm512_loadrs_epi32(
  // CHECK: call <16 x i32> @llvm.x86.avx10.vmovrsd512(
  return _mm512_loadrs_epi32(__A);
}

__m512i test_mm512_mask_loadrs_epi32(__m512i __A, __mmask16 __B, const __m512i * __C) {
  // CHECK-LABEL: @test_mm512_mask_loadrs_epi32(
  // CHECK: call <16 x i32> @llvm.x86.avx10.vmovrsd512(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_loadrs_epi32(__A, __B, __C);
}

__m512i test_mm512_maskz_loadrs_epi32(__mmask16 __A, const __m512i * __B) {
  // CHECK-LABEL: @test_mm512_maskz_loadrs_epi32(
  // CHECK: call <16 x i32> @llvm.x86.avx10.vmovrsd512(
  // CHECK: store <8 x i64> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_loadrs_epi32(__A, __B);
}

__m512i test_mm512_loadrs_epi64(const __m512i * __A) {
  // CHECK-LABEL: @test_mm512_loadrs_epi64(
  // CHECK: call <8 x i64> @llvm.x86.avx10.vmovrsq512(
  return _mm512_loadrs_epi64(__A);
}

__m512i test_mm512_mask_loadrs_epi64(__m512i __A, __mmask8 __B, const __m512i * __C) {
  // CHECK-LABEL: @test_mm512_mask_loadrs_epi64(
  // CHECK: call <8 x i64> @llvm.x86.avx10.vmovrsq512(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_loadrs_epi64(__A, __B, __C);
}

__m512i test_mm512_maskz_loadrs_epi64(__mmask8 __A, const __m512i * __B) {
  // CHECK-LABEL: @test_mm512_maskz_loadrs_epi64(
  // CHECK: call <8 x i64> @llvm.x86.avx10.vmovrsq512(
  // CHECK: store <8 x i64> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_loadrs_epi64(__A, __B);
}

__m512i test_mm512_loadrs_epi16(const __m512i * __A) {
  // CHECK-LABEL: @test_mm512_loadrs_epi16(
  // CHECK: call <32 x i16> @llvm.x86.avx10.vmovrsw512(
  return _mm512_loadrs_epi16(__A);
}

__m512i test_mm512_mask_loadrs_epi16(__m512i __A, __mmask32 __B, const __m512i * __C) {
  // CHECK-LABEL: @test_mm512_mask_loadrs_epi16(
  // CHECK: call <32 x i16> @llvm.x86.avx10.vmovrsw512(
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_loadrs_epi16(__A, __B, __C);
}

__m512i test_mm512_maskz_loadrs_epi16(__mmask32 __A, const __m512i * __B) {
  // CHECK-LABEL: @test_mm512_maskz_loadrs_epi16(
  // CHECK: call <32 x i16> @llvm.x86.avx10.vmovrsw512(
  // CHECK: store <8 x i64> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_loadrs_epi16(__A, __B);
}
