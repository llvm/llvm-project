// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +vaes -emit-llvm -o - | FileCheck %s --check-prefix AVX
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -DAVX512 -target-feature +vaes -target-feature +avx512f -emit-llvm -o - | FileCheck %s --check-prefixes AVX,AVX512

#include <immintrin.h>

__m256i test_mm256_aesenc_epi128(__m256i __A, __m256i __B) {
  // AVX-LABEL: @test_mm256_aesenc_epi128
  // AVX: @llvm.x86.aesni.aesenc.256
  return _mm256_aesenc_epi128(__A, __B);
}

__m256i test_mm256_aesenclast_epi128(__m256i __A, __m256i __B) {
  // AVX-LABEL: @test_mm256_aesenclast_epi128
  // AVX: @llvm.x86.aesni.aesenclast.256
  return _mm256_aesenclast_epi128(__A, __B);
}

__m256i test_mm256_aesdec_epi128(__m256i __A, __m256i __B) {
  // AVX-LABEL: @test_mm256_aesdec_epi128
  // AVX: @llvm.x86.aesni.aesdec.256
  return _mm256_aesdec_epi128(__A, __B);
}

__m256i test_mm256_aesdeclast_epi128(__m256i __A, __m256i __B) {
  // AVX-LABEL: @test_mm256_aesdeclast_epi128
  // AVX: @llvm.x86.aesni.aesdeclast.256
  return _mm256_aesdeclast_epi128(__A, __B);
}

#ifdef AVX512
__m512i test_mm512_aesenc_epi128(__m512i __A, __m512i __B) {
  // AVX512-LABEL: @test_mm512_aesenc_epi128
  // AVX512: @llvm.x86.aesni.aesenc.512
  return _mm512_aesenc_epi128(__A, __B);
}

__m512i test_mm512_aesenclast_epi128(__m512i __A, __m512i __B) {
  // AVX512-LABEL: @test_mm512_aesenclast_epi128
  // AVX512: @llvm.x86.aesni.aesenclast.512
  return _mm512_aesenclast_epi128(__A, __B);
}

__m512i test_mm512_aesdec_epi128(__m512i __A, __m512i __B) {
  // AVX512-LABEL: @test_mm512_aesdec_epi128
  // AVX512: @llvm.x86.aesni.aesdec.512
  return _mm512_aesdec_epi128(__A, __B);
}

__m512i test_mm512_aesdeclast_epi128(__m512i __A, __m512i __B) {
  // AVX512-LABEL: @test_mm512_aesdeclast_epi128
  // AVX512: @llvm.x86.aesni.aesdeclast.512
  return _mm512_aesdeclast_epi128(__A, __B);
}
#endif

