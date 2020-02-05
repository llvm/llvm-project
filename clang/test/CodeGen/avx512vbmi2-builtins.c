// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m512i test_mm512_mask_compress_epi16(__m512i __S, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compress_epi16
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_mask_compress_epi16(__S, __U, __D);
}

__m512i test_mm512_maskz_compress_epi16(__mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_compress_epi16
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_maskz_compress_epi16(__U, __D);
}

__m512i test_mm512_mask_compress_epi8(__m512i __S, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compress_epi8
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_mask_compress_epi8(__S, __U, __D);
}

__m512i test_mm512_maskz_compress_epi8(__mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_compress_epi8
  // CHECK: @llvm.x86.avx512.mask.compress
  return _mm512_maskz_compress_epi8(__U, __D);
}

void test_mm512_mask_compressstoreu_epi16(void *__P, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compressstoreu_epi16
  // CHECK: @llvm.masked.compressstore.v32i16(<32 x i16> %{{.*}}, i16* %{{.*}}, <32 x i1> %{{.*}})
  _mm512_mask_compressstoreu_epi16(__P, __U, __D);
}

void test_mm512_mask_compressstoreu_epi8(void *__P, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_compressstoreu_epi8
  // CHECK: @llvm.masked.compressstore.v64i8(<64 x i8> %{{.*}}, i8* %{{.*}}, <64 x i1> %{{.*}})
  _mm512_mask_compressstoreu_epi8(__P, __U, __D);
}

__m512i test_mm512_mask_expand_epi16(__m512i __S, __mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_expand_epi16
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_mask_expand_epi16(__S, __U, __D);
}

__m512i test_mm512_maskz_expand_epi16(__mmask32 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_expand_epi16
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_maskz_expand_epi16(__U, __D);
}

__m512i test_mm512_mask_expand_epi8(__m512i __S, __mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_mask_expand_epi8
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_mask_expand_epi8(__S, __U, __D);
}

__m512i test_mm512_maskz_expand_epi8(__mmask64 __U, __m512i __D) {
  // CHECK-LABEL: @test_mm512_maskz_expand_epi8
  // CHECK: @llvm.x86.avx512.mask.expand
  return _mm512_maskz_expand_epi8(__U, __D);
}

__m512i test_mm512_mask_expandloadu_epi16(__m512i __S, __mmask32 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_mask_expandloadu_epi16
  // CHECK: @llvm.masked.expandload.v32i16(i16* %{{.*}}, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_mask_expandloadu_epi16(__S, __U, __P);
}

__m512i test_mm512_maskz_expandloadu_epi16(__mmask32 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_maskz_expandloadu_epi16
  // CHECK: @llvm.masked.expandload.v32i16(i16* %{{.*}}, <32 x i1> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_maskz_expandloadu_epi16(__U, __P);
}

__m512i test_mm512_mask_expandloadu_epi8(__m512i __S, __mmask64 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_mask_expandloadu_epi8
  // CHECK: @llvm.masked.expandload.v64i8(i8* %{{.*}}, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_mask_expandloadu_epi8(__S, __U, __P);
}

__m512i test_mm512_maskz_expandloadu_epi8(__mmask64 __U, void const* __P) {
  // CHECK-LABEL: @test_mm512_maskz_expandloadu_epi8
  // CHECK: @llvm.masked.expandload.v64i8(i8* %{{.*}}, <64 x i1> %{{.*}}, <64 x i8> %{{.*}})
  return _mm512_maskz_expandloadu_epi8(__U, __P);
}

__m512i test_mm512_mask_shldi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldi_epi64
  // CHECK: @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> <i64 47, i64 47, i64 47, i64 47, i64 47, i64 47, i64 47, i64 47>)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shldi_epi64(__S, __U, __A, __B, 47);
}

__m512i test_mm512_maskz_shldi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldi_epi64
  // CHECK: @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shldi_epi64(__U, __A, __B, 63);
}

__m512i test_mm512_shldi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldi_epi64
  // CHECK: @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> <i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31>)
  return _mm512_shldi_epi64(__A, __B, 31);
}

__m512i test_mm512_mask_shldi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldi_epi32
  // CHECK: @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shldi_epi32(__S, __U, __A, __B, 7);
}

__m512i test_mm512_maskz_shldi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldi_epi32
  // CHECK: @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shldi_epi32(__U, __A, __B, 15);
}

__m512i test_mm512_shldi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldi_epi32
  // CHECK: @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>)
  return _mm512_shldi_epi32(__A, __B, 31);
}

__m512i test_mm512_mask_shldi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldi_epi16
  // CHECK: @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shldi_epi16(__S, __U, __A, __B, 3);
}

__m512i test_mm512_maskz_shldi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldi_epi16
  // CHECK: @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shldi_epi16(__U, __A, __B, 7);
}

__m512i test_mm512_shldi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldi_epi16
  // CHECK: @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>)
  return _mm512_shldi_epi16(__A, __B, 15);
}

__m512i test_mm512_mask_shrdi_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdi_epi64
  // CHECK: @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> <i64 47, i64 47, i64 47, i64 47, i64 47, i64 47, i64 47, i64 47>)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shrdi_epi64(__S, __U, __A, __B, 47);
}

__m512i test_mm512_maskz_shrdi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdi_epi64
  // CHECK: @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> <i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63, i64 63>)
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shrdi_epi64(__U, __A, __B, 63);
}

__m512i test_mm512_shrdi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdi_epi64
  // CHECK: @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> <i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31, i64 31>)
  return _mm512_shrdi_epi64(__A, __B, 31);
}

__m512i test_mm512_mask_shrdi_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdi_epi32
  // CHECK: @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shrdi_epi32(__S, __U, __A, __B, 7);
}

__m512i test_mm512_maskz_shrdi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdi_epi32
  // CHECK: @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>)
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shrdi_epi32(__U, __A, __B, 15);
}

__m512i test_mm512_shrdi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdi_epi32
  // CHECK: @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>)
  return _mm512_shrdi_epi32(__A, __B, 31);
}

__m512i test_mm512_mask_shrdi_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdi_epi16
  // CHECK: @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shrdi_epi16(__S, __U, __A, __B, 3);
}

__m512i test_mm512_maskz_shrdi_epi16(__mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdi_epi16
  // CHECK: @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>)
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shrdi_epi16(__U, __A, __B, 15);
}

__m512i test_mm512_shrdi_epi16(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdi_epi16
  // CHECK: @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> <i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31, i16 31>)
  return _mm512_shrdi_epi16(__A, __B, 31);
}

__m512i test_mm512_mask_shldv_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldv_epi64
  // CHECK: @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shldv_epi64(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shldv_epi64(__mmask8 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldv_epi64
  // CHECK: @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shldv_epi64(__U, __S, __A, __B);
}

__m512i test_mm512_shldv_epi64(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldv_epi64
  // CHECK: @llvm.fshl.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_shldv_epi64(__S, __A, __B);
}

__m512i test_mm512_mask_shldv_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldv_epi32
  // CHECK: @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shldv_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shldv_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldv_epi32
  // CHECK: @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shldv_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_shldv_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldv_epi32
  // CHECK: @llvm.fshl.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_shldv_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_shldv_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shldv_epi16
  // CHECK: @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shldv_epi16(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shldv_epi16(__mmask32 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shldv_epi16
  // CHECK: @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shldv_epi16(__U, __S, __A, __B);
}

__m512i test_mm512_shldv_epi16(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shldv_epi16
  // CHECK: @llvm.fshl.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_shldv_epi16(__S, __A, __B);
}

__m512i test_mm512_mask_shrdv_epi64(__m512i __S, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdv_epi64
  // CHECK: @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_shrdv_epi64(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shrdv_epi64(__mmask8 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdv_epi64
  // CHECK: @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_shrdv_epi64(__U, __S, __A, __B);
}

__m512i test_mm512_shrdv_epi64(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdv_epi64
  // CHECK: @llvm.fshr.v8i64(<8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}})
  return _mm512_shrdv_epi64(__S, __A, __B);
}

__m512i test_mm512_mask_shrdv_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdv_epi32
  // CHECK: @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_shrdv_epi32(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shrdv_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdv_epi32
  // CHECK: @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_shrdv_epi32(__U, __S, __A, __B);
}

__m512i test_mm512_shrdv_epi32(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdv_epi32
  // CHECK: @llvm.fshr.v16i32(<16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return _mm512_shrdv_epi32(__S, __A, __B);
}

__m512i test_mm512_mask_shrdv_epi16(__m512i __S, __mmask32 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shrdv_epi16
  // CHECK: @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_mask_shrdv_epi16(__S, __U, __A, __B);
}

__m512i test_mm512_maskz_shrdv_epi16(__mmask32 __U, __m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shrdv_epi16
  // CHECK: @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}}
  return _mm512_maskz_shrdv_epi16(__U, __S, __A, __B);
}

__m512i test_mm512_shrdv_epi16(__m512i __S, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shrdv_epi16
  // CHECK: @llvm.fshr.v32i16(<32 x i16> %{{.*}}, <32 x i16> %{{.*}}, <32 x i16> %{{.*}})
  return _mm512_shrdv_epi16(__S, __A, __B);
}

