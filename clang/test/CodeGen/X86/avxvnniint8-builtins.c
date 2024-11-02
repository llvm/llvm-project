// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64- -target-feature +avxvnniint8 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-   -target-feature +avxvnniint8 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64- -target-feature +avx10.2-256 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-   -target-feature +avx10.2-256 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

// CHECK-LABEL: @test_mm_dpbssd_epi32(
// CHECK:     call <4 x i32> @llvm.x86.avx2.vpdpbssd.128
__m128i test_mm_dpbssd_epi32(__m128i __W, __m128i __A, __m128i __B) {
  return _mm_dpbssd_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm_dpbssds_epi32(
// CHECK:     call <4 x i32> @llvm.x86.avx2.vpdpbssds.128
__m128i test_mm_dpbssds_epi32(__m128i __W, __m128i __A, __m128i __B) {
  return _mm_dpbssds_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm_dpbsud_epi32(
// CHECK:     call <4 x i32> @llvm.x86.avx2.vpdpbsud.128
__m128i test_mm_dpbsud_epi32(__m128i __W, __m128i __A, __m128i __B) {
  return _mm_dpbsud_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm_dpbsuds_epi32(
// CHECK:     call <4 x i32> @llvm.x86.avx2.vpdpbsuds.128
__m128i test_mm_dpbsuds_epi32(__m128i __W, __m128i __A, __m128i __B) {
  return _mm_dpbsuds_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm_dpbuud_epi32(
// CHECK:     call <4 x i32> @llvm.x86.avx2.vpdpbuud.128
__m128i test_mm_dpbuud_epi32(__m128i __W, __m128i __A, __m128i __B) {
  return _mm_dpbuud_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm_dpbuuds_epi32(
// CHECK:     call <4 x i32> @llvm.x86.avx2.vpdpbuuds.128
__m128i test_mm_dpbuuds_epi32(__m128i __W, __m128i __A, __m128i __B) {
  return _mm_dpbuuds_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm256_dpbssd_epi32(
// CHECK:     call <8 x i32> @llvm.x86.avx2.vpdpbssd.256
__m256i test_mm256_dpbssd_epi32(__m256i __W, __m256i __A, __m256i __B) {
  return _mm256_dpbssd_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm256_dpbssds_epi32(
// CHECK:     call <8 x i32> @llvm.x86.avx2.vpdpbssds.256
__m256i test_mm256_dpbssds_epi32(__m256i __W, __m256i __A, __m256i __B) {
  return _mm256_dpbssds_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm256_dpbsud_epi32(
// CHECK:     call <8 x i32> @llvm.x86.avx2.vpdpbsud.256
__m256i test_mm256_dpbsud_epi32(__m256i __W, __m256i __A, __m256i __B) {
  return _mm256_dpbsud_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm256_dpbsuds_epi32(
// CHECK:     call <8 x i32> @llvm.x86.avx2.vpdpbsuds.256
__m256i test_mm256_dpbsuds_epi32(__m256i __W, __m256i __A, __m256i __B) {
  return _mm256_dpbsuds_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm256_dpbuud_epi32(
// CHECK:     call <8 x i32> @llvm.x86.avx2.vpdpbuud.256
__m256i test_mm256_dpbuud_epi32(__m256i __W, __m256i __A, __m256i __B) {
  return _mm256_dpbuud_epi32(__W, __A, __B);
}

// CHECK-LABEL: @test_mm256_dpbuuds_epi32(
// CHECK:     call <8 x i32> @llvm.x86.avx2.vpdpbuuds.256
__m256i test_mm256_dpbuuds_epi32(__m256i __W, __m256i __A, __m256i __B) {
  return _mm256_dpbuuds_epi32(__W, __A, __B);
}
