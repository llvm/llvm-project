// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bitalg -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m256i test_mm256_popcnt_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_popcnt_epi16
  // CHECK: @llvm.ctpop.v16i16
  return _mm256_popcnt_epi16(__A);
}

__m256i test_mm256_mask_popcnt_epi16(__m256i __A, __mmask16 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_popcnt_epi16
  // CHECK: @llvm.ctpop.v16i16
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i16> %{{[0-9]+}}, <16 x i16> {{.*}}
  return _mm256_mask_popcnt_epi16(__A, __U, __B);
}
__m256i test_mm256_maskz_popcnt_epi16(__mmask16 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_popcnt_epi16
  // CHECK: @llvm.ctpop.v16i16
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i16> %{{[0-9]+}}, <16 x i16> {{.*}}
  return _mm256_maskz_popcnt_epi16(__U, __B);
}

__m128i test_mm128_popcnt_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm128_popcnt_epi16
  // CHECK: @llvm.ctpop.v8i16
  return _mm128_popcnt_epi16(__A);
}

__m128i test_mm128_mask_popcnt_epi16(__m128i __A, __mmask8 __U, __m128i __B) {
  // CHECK-LABEL: @test_mm128_mask_popcnt_epi16
  // CHECK: @llvm.ctpop.v8i16
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i16> {{.*}}
  return _mm128_mask_popcnt_epi16(__A, __U, __B);
}
__m128i test_mm128_maskz_popcnt_epi16(__mmask8 __U, __m128i __B) {
  // CHECK-LABEL: @test_mm128_maskz_popcnt_epi16
  // CHECK: @llvm.ctpop.v8i16
  // CHECK: select <8 x i1> %{{[0-9]+}}, <8 x i16> %{{[0-9]+}}, <8 x i16> {{.*}}
  return _mm128_maskz_popcnt_epi16(__U, __B);
}

__m256i test_mm256_popcnt_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_popcnt_epi8
  // CHECK: @llvm.ctpop.v32i8
  return _mm256_popcnt_epi8(__A);
}

__m256i test_mm256_mask_popcnt_epi8(__m256i __A, __mmask32 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_popcnt_epi8
  // CHECK: @llvm.ctpop.v32i8
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_mask_popcnt_epi8(__A, __U, __B);
}
__m256i test_mm256_maskz_popcnt_epi8(__mmask32 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_popcnt_epi8
  // CHECK: @llvm.ctpop.v32i8
  // CHECK: select <32 x i1> %{{[0-9]+}}, <32 x i8> %{{[0-9]+}}, <32 x i8> {{.*}}
  return _mm256_maskz_popcnt_epi8(__U, __B);
}

__m128i test_mm128_popcnt_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm128_popcnt_epi8
  // CHECK: @llvm.ctpop.v16i8
  return _mm128_popcnt_epi8(__A);
}

__m128i test_mm128_mask_popcnt_epi8(__m128i __A, __mmask16 __U, __m128i __B) {
  // CHECK-LABEL: @test_mm128_mask_popcnt_epi8
  // CHECK: @llvm.ctpop.v16i8
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm128_mask_popcnt_epi8(__A, __U, __B);
}
__m128i test_mm128_maskz_popcnt_epi8(__mmask16 __U, __m128i __B) {
  // CHECK-LABEL: @test_mm128_maskz_popcnt_epi8
  // CHECK: @llvm.ctpop.v16i8
  // CHECK: select <16 x i1> %{{[0-9]+}}, <16 x i8> %{{[0-9]+}}, <16 x i8> {{.*}}
  return _mm128_maskz_popcnt_epi8(__U, __B);
}

__mmask32 test_mm256_mask_bitshuffle_epi32_mask(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_bitshuffle_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.vpshufbitqmb.256
  return _mm256_mask_bitshuffle_epi32_mask(__U, __A, __B);
}

__mmask32 test_mm256_bitshuffle_epi32_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_bitshuffle_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.vpshufbitqmb.256
  return _mm256_bitshuffle_epi32_mask(__A, __B);
}

__mmask16 test_mm128_mask_bitshuffle_epi16_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm128_mask_bitshuffle_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.vpshufbitqmb.128
  return _mm128_mask_bitshuffle_epi16_mask(__U, __A, __B);
}

__mmask16 test_mm128_bitshuffle_epi16_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm128_bitshuffle_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.vpshufbitqmb.128
  return _mm128_bitshuffle_epi16_mask(__A, __B);
}

