// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_mask_compress_epi16(__m512i __S, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compress_epi16
  // CHECK: @llvm.x86.avx512.mask.compress.w.512
  return _mm512_mask_compress_epi16(__S, __U, __D);
}

__m512i test_mm512_maskz_compress_epi16(__mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_compress_epi16
  // CHECK: @llvm.x86.avx512.mask.compress.w.512
  return _mm512_maskz_compress_epi16(__U, __D);
}

__m512i test_mm512_mask_compress_epi8(__m512i __S, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compress_epi8
  // CHECK: @llvm.x86.avx512.mask.compress.b.512
  return _mm512_mask_compress_epi8(__S, __U, __D);
}

__m512i test_mm512_maskz_compress_epi8(__mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_compress_epi8
  // CHECK: @llvm.x86.avx512.mask.compress.b.512
  return _mm512_maskz_compress_epi8(__U, __D);
}

void test_mm512_mask_compressstoreu_epi16(void *__P, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compressstoreu_epi16
  // CHECK: @llvm.x86.avx512.mask.compress.store.w.512
  _mm512_mask_compressstoreu_epi16(__P, __U, __D);
}

void test_mm512_mask_compressstoreu_epi8(void *__P, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compressstoreu_epi8
  // CHECK: @llvm.x86.avx512.mask.compress.store.b.512
  _mm512_mask_compressstoreu_epi8(__P, __U, __D);
}

__m512i test_mm512_mask_expand_epi16(__m512i __S, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_expand_epi16
  // CHECK: @llvm.x86.avx512.mask.expand.w.512
  return _mm512_mask_expand_epi16(__S, __U, __D);
}

__m512i test_mm512_maskz_expand_epi16(__mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_expand_epi16
  // CHECK: @llvm.x86.avx512.mask.expand.w.512
  return _mm512_maskz_expand_epi16(__U, __D);
}

__m512i test_mm512_mask_expand_epi8(__m512i __S, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_expand_epi8
  // CHECK: @llvm.x86.avx512.mask.expand.b.512
  return _mm512_mask_expand_epi8(__S, __U, __D);
}

__m512i test_mm512_maskz_expand_epi8(__mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_expand_epi8
  // CHECK: @llvm.x86.avx512.mask.expand.b.512
  return _mm512_maskz_expand_epi8(__U, __D);
}

__m512i test_mm512_mask_expandloadu_epi16(__m512i __S, __mmask32 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_mask_expandloadu_epi16
  // CHECK: @llvm.x86.avx512.mask.expand.load.w.512
  return _mm512_mask_expandloadu_epi16(__S, __U, __P);
}

__m512i test_mm512_maskz_expandloadu_epi16(__mmask32 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_maskz_expandloadu_epi16
  // CHECK: @llvm.x86.avx512.mask.expand.load.w.512
  return _mm512_maskz_expandloadu_epi16(__U, __P);
}

__m512i test_mm512_mask_expandloadu_epi8(__m512i __S, __mmask64 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_mask_expandloadu_epi8
  // CHECK: @llvm.x86.avx512.mask.expand.load.b.512
  return _mm512_mask_expandloadu_epi8(__S, __U, __P);
}

__m512i test_mm512_maskz_expandloadu_epi8(__mmask64 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_maskz_expandloadu_epi8
  // CHECK: @llvm.x86.avx512.mask.expand.load.b.512
  return _mm512_maskz_expandloadu_epi8(__U, __P);
}

__m512i test_mm512_mask_shldi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldi_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshld.q.512
  return _mm512_mask_shldi_epi64(__S, __U, __A, __B, 127);
}

__m512i test_mm512_maskz_shldi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldi_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshld.q.512
  return _mm512_maskz_shldi_epi64(__U, __A, __B, 63);
}

__m512i test_mm512_shldi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldi_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshld.q.512
  return _mm512_shldi_epi64(__A, __B, 31);
}

__m512i test_mm512_mask_shldi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldi_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshld.d.512
  return _mm512_mask_shldi_epi32(__S, __U, __A, __B, 127);
}

__m512i test_mm512_maskz_shldi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldi_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshld.d.512
  return _mm512_maskz_shldi_epi32(__U, __A, __B, 63);
}

__m512i test_mm512_shldi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldi_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshld.d.512
  return _mm512_shldi_epi32(__A, __B, 31);
}

__m512i test_mm512_mask_shldi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldi_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshld.w.512
  return _mm512_mask_shldi_epi16(__S, __U, __A, __B, 127);
}

__m512i test_mm512_maskz_shldi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldi_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshld.w.512
  return _mm512_maskz_shldi_epi16(__U, __A, __B, 63);
}

__m512i test_mm512_shldi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldi_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshld.w.512
  return _mm512_shldi_epi16(__A, __B, 31);
}

__m512i test_mm512_mask_shrdi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdi_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshrd.q.512
  return _mm512_mask_shrdi_epi64(__S, __U, __A, __B, 127);
}

__m512i test_mm512_maskz_shrdi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdi_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshrd.q.512
  return _mm512_maskz_shrdi_epi64(__U, __A, __B, 63);
}

__m512i test_mm512_shrdi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdi_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshrd.q.512
  return _mm512_shrdi_epi64(__A, __B, 31);
}

__m512i test_mm512_mask_shrdi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdi_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshrd.d.512
  return _mm512_mask_shrdi_epi32(__S, __U, __A, __B, 127);
}

__m512i test_mm512_maskz_shrdi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdi_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshrd.d.512
  return _mm512_maskz_shrdi_epi32(__U, __A, __B, 63);
}

__m512i test_mm512_shrdi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdi_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshrd.d.512
  return _mm512_shrdi_epi32(__A, __B, 31);
}

__m512i test_mm512_mask_shrdi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdi_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshrd.w.512
  return _mm512_mask_shrdi_epi16(__S, __U, __A, __B, 127);
}

__m512i test_mm512_maskz_shrdi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdi_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshrd.w.512
  return _mm512_maskz_shrdi_epi16(__U, __A, __B, 63);
}

__m512i test_mm512_shrdi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdi_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshrd.w.512
  return _mm512_shrdi_epi16(__A, __B, 31);
}

__m512i test_mm512_mask_shldv_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldv_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshldv.q.512
  return _mm512_mask_shldv_epi64(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shldv_epi64(__mmask8 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldv_epi64
  // CHECK: @llvm.x86.avx512.maskz.vpshldv.q.512
  return _mm512_maskz_shldv_epi64(__U, __S, __A, __B);
}

__m512i test_mm512_shldv_epi64(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldv_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshldv.q.512
  return _mm512_shldv_epi64(__S, __A, __B);
}

__m512i test_mm512_mask_shldv_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldv_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshldv.d.512
  return _mm512_mask_shldv_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shldv_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldv_epi32
  // CHECK: @llvm.x86.avx512.maskz.vpshldv.d.512
  return _mm512_maskz_shldv_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_shldv_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldv_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshldv.d.512
  return _mm512_shldv_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_shldv_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldv_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshldv.w.512
  return _mm512_mask_shldv_epi16(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shldv_epi16(__mmask32 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldv_epi16
  // CHECK: @llvm.x86.avx512.maskz.vpshldv.w.512
  return _mm512_maskz_shldv_epi16(__U, __S, __A, __B);
}

__m512i test_mm512_shldv_epi16(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldv_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshldv.w.512
  return _mm512_shldv_epi16(__S, __A, __B);
}

__m512i test_mm512_mask_shrdv_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdv_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshrdv.q.512
  return _mm512_mask_shrdv_epi64(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shrdv_epi64(__mmask8 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdv_epi64
  // CHECK: @llvm.x86.avx512.maskz.vpshrdv.q.512
  return _mm512_maskz_shrdv_epi64(__U, __S, __A, __B);
}

__m512i test_mm512_shrdv_epi64(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdv_epi64
  // CHECK: @llvm.x86.avx512.mask.vpshrdv.q.512
  return _mm512_shrdv_epi64(__S, __A, __B);
}

__m512i test_mm512_mask_shrdv_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdv_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshrdv.d.512
  return _mm512_mask_shrdv_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shrdv_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdv_epi32
  // CHECK: @llvm.x86.avx512.maskz.vpshrdv.d.512
  return _mm512_maskz_shrdv_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_shrdv_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdv_epi32
  // CHECK: @llvm.x86.avx512.mask.vpshrdv.d.512
  return _mm512_shrdv_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_shrdv_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdv_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshrdv.w.512
  return _mm512_mask_shrdv_epi16(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shrdv_epi16(__mmask32 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdv_epi16
  // CHECK: @llvm.x86.avx512.maskz.vpshrdv.w.512
  return _mm512_maskz_shrdv_epi16(__U, __S, __A, __B);
}

__m512i test_mm512_shrdv_epi16(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdv_epi16
  // CHECK: @llvm.x86.avx512.mask.vpshrdv.w.512
  return _mm512_shrdv_epi16(__S, __A, __B);
}

