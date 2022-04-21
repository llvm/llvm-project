// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxvnni -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxvnni -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m256i test_mm256_dpbusd_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpbusd_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusd.256
  return _mm256_dpbusd_epi32(__S, __A, __B);
}

__m256i test_mm256_dpbusds_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpbusds_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusds.256
  return _mm256_dpbusds_epi32(__S, __A, __B);
}

__m256i test_mm256_dpwssd_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpwssd_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssd.256
  return _mm256_dpwssd_epi32(__S, __A, __B);
}

__m256i test_mm256_dpwssds_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpwssds_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssds.256
  return _mm256_dpwssds_epi32(__S, __A, __B);
}

__m128i test_mm_dpbusd_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpbusd_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusd.128
  return _mm_dpbusd_epi32(__S, __A, __B);
}

__m128i test_mm_dpbusds_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpbusds_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusds.128
  return _mm_dpbusds_epi32(__S, __A, __B);
}

__m128i test_mm_dpwssd_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpwssd_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssd.128
  return _mm_dpwssd_epi32(__S, __A, __B);
}

__m128i test_mm_dpwssds_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpwssds_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssds.128
  return _mm_dpwssds_epi32(__S, __A, __B);
}

__m256i test_mm256_dpbusd_avx_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpbusd_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusd.256
  return _mm256_dpbusd_avx_epi32(__S, __A, __B);
}

__m256i test_mm256_dpbusds_avx_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpbusds_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusds.256
  return _mm256_dpbusds_avx_epi32(__S, __A, __B);
}

__m256i test_mm256_dpwssd_avx_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpwssd_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssd.256
  return _mm256_dpwssd_avx_epi32(__S, __A, __B);
}

__m256i test_mm256_dpwssds_avx_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dpwssds_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssds.256
  return _mm256_dpwssds_avx_epi32(__S, __A, __B);
}

__m128i test_mm_dpbusd_avx_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpbusd_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusd.128
  return _mm_dpbusd_avx_epi32(__S, __A, __B);
}

__m128i test_mm_dpbusds_avx_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpbusds_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpbusds.128
  return _mm_dpbusds_avx_epi32(__S, __A, __B);
}

__m128i test_mm_dpwssd_avx_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpwssd_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssd.128
  return _mm_dpwssd_avx_epi32(__S, __A, __B);
}

__m128i test_mm_dpwssds_avx_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dpwssds_avx_epi32
  // CHECK: @llvm.x86.avx512.vpdpwssds.128
  return _mm_dpwssds_avx_epi32(__S, __A, __B);
}
