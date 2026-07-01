// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64 -target-feature +avx10-v2-aux \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=i386 -target-feature +avx10-v2-aux \
// RUN: -emit-llvm -o - -Wno-invalid-feature-combination -Wall -Werror | FileCheck %s

#include <immintrin.h>

//
// Group A: VCVTPS2BF8 / VCVTPS2BF8S / VCVTPS2HF8 / VCVTPS2HF8S /
//          VCVTROPS2HF8 / VCVTROPS2HF8S
//

// VCVTPS2BF8 - 128-bit

__m128i test_mm_cvtps_bf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.128(
  return _mm_cvtps_bf8(__A);
}

__m128i test_mm_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.128(
  return _mm_mask_cvtps_bf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtps_bf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.128(
  return _mm_maskz_cvtps_bf8(__U, __A);
}

// VCVTPS2BF8 - 256-bit

__m128i test_mm256_cvtps_bf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.256(
  return _mm256_cvtps_bf8(__A);
}

__m128i test_mm256_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.256(
  return _mm256_mask_cvtps_bf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtps_bf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.256(
  return _mm256_maskz_cvtps_bf8(__U, __A);
}

// VCVTPS2BF8 - 512-bit

__m128i test_mm512_cvtps_bf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.512(
  return _mm512_cvtps_bf8(__A);
}

__m128i test_mm512_mask_cvtps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.512(
  return _mm512_mask_cvtps_bf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtps_bf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8.512(
  return _mm512_maskz_cvtps_bf8(__U, __A);
}

// VCVTPS2BF8S - 128-bit

__m128i test_mm_cvts_ps_bf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.128(
  return _mm_cvts_ps_bf8(__A);
}

__m128i test_mm_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.128(
  return _mm_mask_cvts_ps_bf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvts_ps_bf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.128(
  return _mm_maskz_cvts_ps_bf8(__U, __A);
}

// VCVTPS2BF8S - 256-bit

__m128i test_mm256_cvts_ps_bf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.256(
  return _mm256_cvts_ps_bf8(__A);
}

__m128i test_mm256_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.256(
  return _mm256_mask_cvts_ps_bf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvts_ps_bf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.256(
  return _mm256_maskz_cvts_ps_bf8(__U, __A);
}

// VCVTPS2BF8S - 512-bit

__m128i test_mm512_cvts_ps_bf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.512(
  return _mm512_cvts_ps_bf8(__A);
}

__m128i test_mm512_mask_cvts_ps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.512(
  return _mm512_mask_cvts_ps_bf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvts_ps_bf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_ps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2bf8s.512(
  return _mm512_maskz_cvts_ps_bf8(__U, __A);
}

// VCVTPS2HF8 - 128-bit

__m128i test_mm_cvtps_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.128(
  return _mm_cvtps_hf8(__A);
}

__m128i test_mm_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.128(
  return _mm_mask_cvtps_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtps_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.128(
  return _mm_maskz_cvtps_hf8(__U, __A);
}

// VCVTPS2HF8 - 256-bit

__m128i test_mm256_cvtps_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.256(
  return _mm256_cvtps_hf8(__A);
}

__m128i test_mm256_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.256(
  return _mm256_mask_cvtps_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtps_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.256(
  return _mm256_maskz_cvtps_hf8(__U, __A);
}

// VCVTPS2HF8 - 512-bit

__m128i test_mm512_cvtps_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.512(
  return _mm512_cvtps_hf8(__A);
}

__m128i test_mm512_mask_cvtps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.512(
  return _mm512_mask_cvtps_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtps_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8.512(
  return _mm512_maskz_cvtps_hf8(__U, __A);
}

// VCVTPS2HF8S - 128-bit

__m128i test_mm_cvts_ps_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.128(
  return _mm_cvts_ps_hf8(__A);
}

__m128i test_mm_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.128(
  return _mm_mask_cvts_ps_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvts_ps_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.128(
  return _mm_maskz_cvts_ps_hf8(__U, __A);
}

// VCVTPS2HF8S - 256-bit

__m128i test_mm256_cvts_ps_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.256(
  return _mm256_cvts_ps_hf8(__A);
}

__m128i test_mm256_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.256(
  return _mm256_mask_cvts_ps_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvts_ps_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.256(
  return _mm256_maskz_cvts_ps_hf8(__U, __A);
}

// VCVTPS2HF8S - 512-bit

__m128i test_mm512_cvts_ps_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.512(
  return _mm512_cvts_ps_hf8(__A);
}

__m128i test_mm512_mask_cvts_ps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.512(
  return _mm512_mask_cvts_ps_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvts_ps_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_ps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtps2hf8s.512(
  return _mm512_maskz_cvts_ps_hf8(__U, __A);
}

// VCVTROPS2HF8 - 128-bit

__m128i test_mm_cvtrops_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.128(
  return _mm_cvtrops_hf8(__A);
}

__m128i test_mm_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.128(
  return _mm_mask_cvtrops_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtrops_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.128(
  return _mm_maskz_cvtrops_hf8(__U, __A);
}

// VCVTROPS2HF8 - 256-bit

__m128i test_mm256_cvtrops_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.256(
  return _mm256_cvtrops_hf8(__A);
}

__m128i test_mm256_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.256(
  return _mm256_mask_cvtrops_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtrops_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.256(
  return _mm256_maskz_cvtrops_hf8(__U, __A);
}

// VCVTROPS2HF8 - 512-bit

__m128i test_mm512_cvtrops_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.512(
  return _mm512_cvtrops_hf8(__A);
}

__m128i test_mm512_mask_cvtrops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.512(
  return _mm512_mask_cvtrops_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvtrops_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtrops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8.512(
  return _mm512_maskz_cvtrops_hf8(__U, __A);
}

// VCVTROPS2HF8S - 128-bit

__m128i test_mm_cvts_rops_hf8(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.128(
  return _mm_cvts_rops_hf8(__A);
}

__m128i test_mm_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.128(
  return _mm_mask_cvts_rops_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvts_rops_hf8(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.128(
  return _mm_maskz_cvts_rops_hf8(__U, __A);
}

// VCVTROPS2HF8S - 256-bit

__m128i test_mm256_cvts_rops_hf8(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.256(
  return _mm256_cvts_rops_hf8(__A);
}

__m128i test_mm256_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.256(
  return _mm256_mask_cvts_rops_hf8(__W, __U, __A);
}

__m128i test_mm256_maskz_cvts_rops_hf8(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.256(
  return _mm256_maskz_cvts_rops_hf8(__U, __A);
}

// VCVTROPS2HF8S - 512-bit

__m128i test_mm512_cvts_rops_hf8(__m512 __A) {
  // CHECK-LABEL: @test_mm512_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.512(
  return _mm512_cvts_rops_hf8(__A);
}

__m128i test_mm512_mask_cvts_rops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.512(
  return _mm512_mask_cvts_rops_hf8(__W, __U, __A);
}

__m128i test_mm512_maskz_cvts_rops_hf8(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_rops_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtrops2hf8s.512(
  return _mm512_maskz_cvts_rops_hf8(__U, __A);
}

//
// Group B: VCVTBIASPS2BF8 / VCVTBIASPS2BF8S / VCVTBIASPS2HF8 /
//          VCVTBIASPS2HF8S
//

// VCVTBIASPS2BF8 - 128-bit

__m128i test_mm_cvtbiasps_bf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.128(
  return _mm_cvtbiasps_bf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.128(
  return _mm_mask_cvtbiasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.128(
  return _mm_maskz_cvtbiasps_bf8(__U, __A, __B);
}

// VCVTBIASPS2BF8 - 256-bit

__m128i test_mm256_cvtbiasps_bf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.256(
  return _mm256_cvtbiasps_bf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.256(
  return _mm256_mask_cvtbiasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.256(
  return _mm256_maskz_cvtbiasps_bf8(__U, __A, __B);
}

// VCVTBIASPS2BF8 - 512-bit

__m128i test_mm512_cvtbiasps_bf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.512(
  return _mm512_cvtbiasps_bf8(__A, __B);
}

__m128i test_mm512_mask_cvtbiasps_bf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.512(
  return _mm512_mask_cvtbiasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvtbiasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8.512(
  return _mm512_maskz_cvtbiasps_bf8(__U, __A, __B);
}

// VCVTBIASPS2BF8S - 128-bit

__m128i test_mm_cvts_biasps_bf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.128(
  return _mm_cvts_biasps_bf8(__A, __B);
}

__m128i test_mm_mask_cvts_biasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.128(
  return _mm_mask_cvts_biasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvts_biasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.128(
  return _mm_maskz_cvts_biasps_bf8(__U, __A, __B);
}

// VCVTBIASPS2BF8S - 256-bit

__m128i test_mm256_cvts_biasps_bf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.256(
  return _mm256_cvts_biasps_bf8(__A, __B);
}

__m128i test_mm256_mask_cvts_biasps_bf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.256(
  return _mm256_mask_cvts_biasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvts_biasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.256(
  return _mm256_maskz_cvts_biasps_bf8(__U, __A, __B);
}

// VCVTBIASPS2BF8S - 512-bit

__m128i test_mm512_cvts_biasps_bf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.512(
  return _mm512_cvts_biasps_bf8(__A, __B);
}

__m128i test_mm512_mask_cvts_biasps_bf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.512(
  return _mm512_mask_cvts_biasps_bf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvts_biasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_biasps_bf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2bf8s.512(
  return _mm512_maskz_cvts_biasps_bf8(__U, __A, __B);
}

// VCVTBIASPS2HF8 - 128-bit

__m128i test_mm_cvtbiasps_hf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.128(
  return _mm_cvtbiasps_hf8(__A, __B);
}

__m128i test_mm_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.128(
  return _mm_mask_cvtbiasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvtbiasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.128(
  return _mm_maskz_cvtbiasps_hf8(__U, __A, __B);
}

// VCVTBIASPS2HF8 - 256-bit

__m128i test_mm256_cvtbiasps_hf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.256(
  return _mm256_cvtbiasps_hf8(__A, __B);
}

__m128i test_mm256_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.256(
  return _mm256_mask_cvtbiasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvtbiasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.256(
  return _mm256_maskz_cvtbiasps_hf8(__U, __A, __B);
}

// VCVTBIASPS2HF8 - 512-bit

__m128i test_mm512_cvtbiasps_hf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.512(
  return _mm512_cvtbiasps_hf8(__A, __B);
}

__m128i test_mm512_mask_cvtbiasps_hf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.512(
  return _mm512_mask_cvtbiasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvtbiasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbiasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8.512(
  return _mm512_maskz_cvtbiasps_hf8(__U, __A, __B);
}

// VCVTBIASPS2HF8S - 128-bit

__m128i test_mm_cvts_biasps_hf8(__m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.128(
  return _mm_cvts_biasps_hf8(__A, __B);
}

__m128i test_mm_mask_cvts_biasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.128(
  return _mm_mask_cvts_biasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm_maskz_cvts_biasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.128(
  return _mm_maskz_cvts_biasps_hf8(__U, __A, __B);
}

// VCVTBIASPS2HF8S - 256-bit

__m128i test_mm256_cvts_biasps_hf8(__m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.256(
  return _mm256_cvts_biasps_hf8(__A, __B);
}

__m128i test_mm256_mask_cvts_biasps_hf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.256(
  return _mm256_mask_cvts_biasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm256_maskz_cvts_biasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.256(
  return _mm256_maskz_cvts_biasps_hf8(__U, __A, __B);
}

// VCVTBIASPS2HF8S - 512-bit

__m128i test_mm512_cvts_biasps_hf8(__m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.512(
  return _mm512_cvts_biasps_hf8(__A, __B);
}

__m128i test_mm512_mask_cvts_biasps_hf8(__m128i __W, __mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.512(
  return _mm512_mask_cvts_biasps_hf8(__W, __U, __A, __B);
}

__m128i test_mm512_maskz_cvts_biasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_cvts_biasps_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.mask.vcvtbiasps2hf8s.512(
  return _mm512_maskz_cvts_biasps_hf8(__U, __A, __B);
}

//
// Group C: VCVTBF82PS / VCVTHF82PS
//

// VCVTBF82PS - 128-bit

__m128 test_mm_cvtbf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vcvtbf82ps.128(
  return _mm_cvtbf8_ps(__A);
}

__m128 test_mm_mask_cvtbf8_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vcvtbf82ps.128(
  return _mm_mask_cvtbf8_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vcvtbf82ps.128(
  return _mm_maskz_cvtbf8_ps(__U, __A);
}

// VCVTBF82PS - 256-bit

__m256 test_mm256_cvtbf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vcvtbf82ps.256(
  return _mm256_cvtbf8_ps(__A);
}

__m256 test_mm256_mask_cvtbf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vcvtbf82ps.256(
  return _mm256_mask_cvtbf8_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vcvtbf82ps.256(
  return _mm256_maskz_cvtbf8_ps(__U, __A);
}

// VCVTBF82PS - 512-bit

__m512 test_mm512_cvtbf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vcvtbf82ps.512(
  return _mm512_cvtbf8_ps(__A);
}

__m512 test_mm512_mask_cvtbf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtbf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vcvtbf82ps.512(
  return _mm512_mask_cvtbf8_ps(__W, __U, __A);
}

__m512 test_mm512_maskz_cvtbf8_ps(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vcvtbf82ps.512(
  return _mm512_maskz_cvtbf8_ps(__U, __A);
}

// VCVTHF82PS - 128-bit

__m128 test_mm_cvthf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvthf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vcvthf82ps.128(
  return _mm_cvthf8_ps(__A);
}

__m128 test_mm_mask_cvthf8_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvthf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vcvthf82ps.128(
  return _mm_mask_cvthf8_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvthf8_ps(
  // CHECK: call <4 x float> @llvm.x86.avx10.mask.vcvthf82ps.128(
  return _mm_maskz_cvthf8_ps(__U, __A);
}

// VCVTHF82PS - 256-bit

__m256 test_mm256_cvthf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvthf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vcvthf82ps.256(
  return _mm256_cvthf8_ps(__A);
}

__m256 test_mm256_mask_cvthf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvthf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vcvthf82ps.256(
  return _mm256_mask_cvthf8_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvthf8_ps(
  // CHECK: call <8 x float> @llvm.x86.avx10.mask.vcvthf82ps.256(
  return _mm256_maskz_cvthf8_ps(__U, __A);
}

// VCVTHF82PS - 512-bit

__m512 test_mm512_cvthf8_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvthf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vcvthf82ps.512(
  return _mm512_cvthf8_ps(__A);
}

__m512 test_mm512_mask_cvthf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvthf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vcvthf82ps.512(
  return _mm512_mask_cvthf8_ps(__W, __U, __A);
}

__m512 test_mm512_maskz_cvthf8_ps(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvthf8_ps(
  // CHECK: call <16 x float> @llvm.x86.avx10.mask.vcvthf82ps.512(
  return _mm512_maskz_cvthf8_ps(__U, __A);
}

//
// Group E: VCVTBF82BF6S / VCVTHF82HF6S
//

// VCVTBF82BF6S

__m128i test_mm_cvtbf8_bf6s(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf8_bf6s(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf82bf6s.128(
  return _mm_cvtbf8_bf6s(__A);
}

__m256i test_mm256_cvtbf8_bf6s(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf8_bf6s(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf82bf6s.256(
  return _mm256_cvtbf8_bf6s(__A);
}

__m512i test_mm512_cvtbf8_bf6s(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf8_bf6s(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf82bf6s.512(
  return _mm512_cvtbf8_bf6s(__A);
}

// VCVTHF82HF6S

__m128i test_mm_cvthf8_hf6s(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvthf8_hf6s(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf82hf6s.128(
  return _mm_cvthf8_hf6s(__A);
}

__m256i test_mm256_cvthf8_hf6s(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvthf8_hf6s(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf82hf6s.256(
  return _mm256_cvthf8_hf6s(__A);
}

__m512i test_mm512_cvthf8_hf6s(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvthf8_hf6s(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvthf82hf6s.512(
  return _mm512_cvthf8_hf6s(__A);
}

//
// Group F: VCVTBF42HF8 / VCVTBF62HF8 / VCVTHF62HF8
//

// VCVTBF42HF8 - 128-bit

__m128i test_mm_cvtbf4_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf4_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf42hf8.128(
  return _mm_cvtbf4_hf8(__A);
}

__m128i test_mm_mask_cvtbf4_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf4_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf42hf8.128(
  // CHECK: select <16 x i1>
  return _mm_mask_cvtbf4_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtbf4_hf8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf4_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf42hf8.128(
  // CHECK: select <16 x i1>
  return _mm_maskz_cvtbf4_hf8(__U, __A);
}

// VCVTBF42HF8 - 256-bit

__m256i test_mm256_cvtbf4_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf4_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf42hf8.256(
  return _mm256_cvtbf4_hf8(__A);
}

__m256i test_mm256_mask_cvtbf4_hf8(__m256i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf4_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf42hf8.256(
  // CHECK: select <32 x i1>
  return _mm256_mask_cvtbf4_hf8(__W, __U, __A);
}

__m256i test_mm256_maskz_cvtbf4_hf8(__mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf4_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf42hf8.256(
  // CHECK: select <32 x i1>
  return _mm256_maskz_cvtbf4_hf8(__U, __A);
}

// VCVTBF42HF8 - 512-bit

__m512i test_mm512_cvtbf4_hf8(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf4_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf42hf8.512(
  return _mm512_cvtbf4_hf8(__A);
}

__m512i test_mm512_mask_cvtbf4_hf8(__m512i __W, __mmask64 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtbf4_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf42hf8.512(
  // CHECK: select <64 x i1>
  return _mm512_mask_cvtbf4_hf8(__W, __U, __A);
}

__m512i test_mm512_maskz_cvtbf4_hf8(__mmask64 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbf4_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf42hf8.512(
  // CHECK: select <64 x i1>
  return _mm512_maskz_cvtbf4_hf8(__U, __A);
}

// VCVTBF62HF8 - 128-bit

__m128i test_mm_cvtbf6_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtbf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf62hf8.128(
  return _mm_cvtbf6_hf8(__A);
}

__m128i test_mm_mask_cvtbf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtbf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf62hf8.128(
  // CHECK: select <16 x i1>
  return _mm_mask_cvtbf6_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvtbf6_hf8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtbf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvtbf62hf8.128(
  // CHECK: select <16 x i1>
  return _mm_maskz_cvtbf6_hf8(__U, __A);
}

// VCVTBF62HF8 - 256-bit

__m256i test_mm256_cvtbf6_hf8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtbf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf62hf8.256(
  return _mm256_cvtbf6_hf8(__A);
}

__m256i test_mm256_mask_cvtbf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtbf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf62hf8.256(
  // CHECK: select <32 x i1>
  return _mm256_mask_cvtbf6_hf8(__W, __U, __A);
}

__m256i test_mm256_maskz_cvtbf6_hf8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtbf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvtbf62hf8.256(
  // CHECK: select <32 x i1>
  return _mm256_maskz_cvtbf6_hf8(__U, __A);
}

// VCVTBF62HF8 - 512-bit

__m512i test_mm512_cvtbf6_hf8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtbf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf62hf8.512(
  return _mm512_cvtbf6_hf8(__A);
}

__m512i test_mm512_mask_cvtbf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtbf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf62hf8.512(
  // CHECK: select <64 x i1>
  return _mm512_mask_cvtbf6_hf8(__W, __U, __A);
}

__m512i test_mm512_maskz_cvtbf6_hf8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtbf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvtbf62hf8.512(
  // CHECK: select <64 x i1>
  return _mm512_maskz_cvtbf6_hf8(__U, __A);
}

// VCVTHF62HF8 - 128-bit

__m128i test_mm_cvthf6_hf8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvthf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf62hf8.128(
  return _mm_cvthf6_hf8(__A);
}

__m128i test_mm_mask_cvthf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvthf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf62hf8.128(
  // CHECK: select <16 x i1>
  return _mm_mask_cvthf6_hf8(__W, __U, __A);
}

__m128i test_mm_maskz_cvthf6_hf8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvthf6_hf8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vcvthf62hf8.128(
  // CHECK: select <16 x i1>
  return _mm_maskz_cvthf6_hf8(__U, __A);
}

// VCVTHF62HF8 - 256-bit

__m256i test_mm256_cvthf6_hf8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvthf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf62hf8.256(
  return _mm256_cvthf6_hf8(__A);
}

__m256i test_mm256_mask_cvthf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvthf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf62hf8.256(
  // CHECK: select <32 x i1>
  return _mm256_mask_cvthf6_hf8(__W, __U, __A);
}

__m256i test_mm256_maskz_cvthf6_hf8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvthf6_hf8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vcvthf62hf8.256(
  // CHECK: select <32 x i1>
  return _mm256_maskz_cvthf6_hf8(__U, __A);
}

// VCVTHF62HF8 - 512-bit

__m512i test_mm512_cvthf6_hf8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvthf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvthf62hf8.512(
  return _mm512_cvthf6_hf8(__A);
}

__m512i test_mm512_mask_cvthf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvthf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvthf62hf8.512(
  // CHECK: select <64 x i1>
  return _mm512_mask_cvthf6_hf8(__W, __U, __A);
}

__m512i test_mm512_maskz_cvthf6_hf8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvthf6_hf8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vcvthf62hf8.512(
  // CHECK: select <64 x i1>
  return _mm512_maskz_cvthf6_hf8(__U, __A);
}

//
// Group H: VUNPACKB
//

// VUNPACKB - 128-bit

__m128i test_mm_unpackb_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_unpackb_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(
  return _mm_unpackb_epi8(__A, 1);
}

__m128i test_mm_mask_unpackb_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_unpackb_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(
  // CHECK: select <16 x i1>
  return _mm_mask_unpackb_epi8(__W, __U, __A, 1);
}

__m128i test_mm_maskz_unpackb_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_unpackb_epi8(
  // CHECK: call <16 x i8> @llvm.x86.avx10.vunpackb.128(
  // CHECK: select <16 x i1>
  return _mm_maskz_unpackb_epi8(__U, __A, 1);
}

// VUNPACKB - 256-bit

__m256i test_mm256_unpackb_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_unpackb_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(
  return _mm256_unpackb_epi8(__A, 2);
}

__m256i test_mm256_mask_unpackb_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_unpackb_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(
  // CHECK: select <32 x i1>
  return _mm256_mask_unpackb_epi8(__W, __U, __A, 2);
}

__m256i test_mm256_maskz_unpackb_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_unpackb_epi8(
  // CHECK: call <32 x i8> @llvm.x86.avx10.vunpackb.256(
  // CHECK: select <32 x i1>
  return _mm256_maskz_unpackb_epi8(__U, __A, 2);
}

// VUNPACKB - 512-bit

__m512i test_mm512_unpackb_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_unpackb_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(
  return _mm512_unpackb_epi8(__A, 3);
}

__m512i test_mm512_mask_unpackb_epi8(__m512i __W, __mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_unpackb_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(
  // CHECK: select <64 x i1>
  return _mm512_mask_unpackb_epi8(__W, __U, __A, 3);
}

__m512i test_mm512_maskz_unpackb_epi8(__mmask64 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_unpackb_epi8(
  // CHECK: call <64 x i8> @llvm.x86.avx10.vunpackb.512(
  // CHECK: select <64 x i1>
  return _mm512_maskz_unpackb_epi8(__U, __A, 3);
}
