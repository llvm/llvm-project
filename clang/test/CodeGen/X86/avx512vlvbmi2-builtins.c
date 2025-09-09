// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__m128i test_mm_mask_compress_epi16(__m128i __S, __mmask8 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_mask_compress_epi16
  // CHECK: call <8 x i16> @llvm.x86.avx512.mask.compress.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i1> %{{.*}})
  return _mm_mask_compress_epi16(__S, __U, __D);
}

__m128i test_mm_maskz_compress_epi16(__mmask8 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_maskz_compress_epi16
  // CHECK: call <8 x i16> @llvm.x86.avx512.mask.compress.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i1> %{{.*}})
  return _mm_maskz_compress_epi16(__U, __D);
}

__m128i test_mm_mask_compress_epi8(__m128i __S, __mmask16 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_mask_compress_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.mask.compress.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i1> %{{.*}})
  return _mm_mask_compress_epi8(__S, __U, __D);
}

__m128i test_mm_maskz_compress_epi8(__mmask16 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_maskz_compress_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.mask.compress.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i1> %{{.*}})
  return _mm_maskz_compress_epi8(__U, __D);
}

void test_mm_mask_compressstoreu_epi16(void *__P, __mmask8 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_mask_compressstoreu_epi16
  // CHECK: call void @llvm.masked.compressstore.v8i16(<8 x i16> %{{.*}}, ptr %{{.*}}, <8 x i1> %{{.*}})
  _mm_mask_compressstoreu_epi16(__P, __U, __D);
}

void test_mm_mask_compressstoreu_epi8(void *__P, __mmask16 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_mask_compressstoreu_epi8
  // CHECK: call void @llvm.masked.compressstore.v16i8(<16 x i8> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  _mm_mask_compressstoreu_epi8(__P, __U, __D);
}

__m128i test_mm_mask_expand_epi16(__m128i __S, __mmask8 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_mask_expand_epi16
  // CHECK: call <8 x i16> @llvm.x86.avx512.mask.expand.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i1> %{{.*}})
  return _mm_mask_expand_epi16(__S, __U, __D);
}

__m128i test_mm_maskz_expand_epi16(__mmask8 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_maskz_expand_epi16
  // CHECK: call <8 x i16> @llvm.x86.avx512.mask.expand.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i1> %{{.*}})
  return _mm_maskz_expand_epi16(__U, __D);
}

__m128i test_mm_mask_expand_epi8(__m128i __S, __mmask16 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_mask_expand_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.mask.expand.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i1> %{{.*}})
  return _mm_mask_expand_epi8(__S, __U, __D);
}

__m128i test_mm_maskz_expand_epi8(__mmask16 __U, __m128i __D) {
  // CHECK-LABEL: test_mm_maskz_expand_epi8
  // CHECK: call <16 x i8> @llvm.x86.avx512.mask.expand.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i1> %{{.*}})
  return _mm_maskz_expand_epi8(__U, __D);
}

__m128i test_mm_mask_expandloadu_epi16(__m128i __S, __mmask8 __U, void const* __P) {
  // CHECK-LABEL: test_mm_mask_expandloadu_epi16
  // CHECK: call <8 x i16> @llvm.masked.expandload.v8i16(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mask_expandloadu_epi16(__S, __U, __P);
}

__m128i test_mm_maskz_expandloadu_epi16(__mmask8 __U, void const* __P) {
  // CHECK-LABEL: test_mm_maskz_expandloadu_epi16
  // CHECK: call <8 x i16> @llvm.masked.expandload.v8i16(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_maskz_expandloadu_epi16(__U, __P);
}

__m128i test_mm_mask_expandloadu_epi8(__m128i __S, __mmask16 __U, void const* __P) {
  // CHECK-LABEL: test_mm_mask_expandloadu_epi8
  // CHECK: call <16 x i8> @llvm.masked.expandload.v16i8(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_mask_expandloadu_epi8(__S, __U, __P);
}

__m128i test_mm_maskz_expandloadu_epi8(__mmask16 __U, void const* __P) {
  // CHECK-LABEL: test_mm_maskz_expandloadu_epi8
  // CHECK: call <16 x i8> @llvm.masked.expandload.v16i8(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_maskz_expandloadu_epi8(__U, __P);
}

__m256i test_mm256_mask_compress_epi16(__m256i __S, __mmask16 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_mask_compress_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx512.mask.compress.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i1> %{{.*}})
  return _mm256_mask_compress_epi16(__S, __U, __D);
}

__m256i test_mm256_maskz_compress_epi16(__mmask16 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_maskz_compress_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx512.mask.compress.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i1> %{{.*}})
  return _mm256_maskz_compress_epi16(__U, __D);
}

__m256i test_mm256_mask_compress_epi8(__m256i __S, __mmask32 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_mask_compress_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.mask.compress.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i1> %{{.*}})
  return _mm256_mask_compress_epi8(__S, __U, __D);
}

__m256i test_mm256_maskz_compress_epi8(__mmask32 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_maskz_compress_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.mask.compress.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i1> %{{.*}})
  return _mm256_maskz_compress_epi8(__U, __D);
}

void test_mm256_mask_compressstoreu_epi16(void *__P, __mmask16 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_mask_compressstoreu_epi16
  // CHECK: call void @llvm.masked.compressstore.v16i16(<16 x i16> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  _mm256_mask_compressstoreu_epi16(__P, __U, __D);
}

void test_mm256_mask_compressstoreu_epi8(void *__P, __mmask32 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_mask_compressstoreu_epi8
  // CHECK: call void @llvm.masked.compressstore.v32i8(<32 x i8> %{{.*}}, ptr %{{.*}}, <32 x i1> %{{.*}})
  _mm256_mask_compressstoreu_epi8(__P, __U, __D);
}

__m256i test_mm256_mask_expand_epi16(__m256i __S, __mmask16 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_mask_expand_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx512.mask.expand.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i1> %{{.*}})
  return _mm256_mask_expand_epi16(__S, __U, __D);
}

__m256i test_mm256_maskz_expand_epi16(__mmask16 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_maskz_expand_epi16
  // CHECK: call <16 x i16> @llvm.x86.avx512.mask.expand.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i1> %{{.*}})
  return _mm256_maskz_expand_epi16(__U, __D);
}

__m256i test_mm256_mask_expand_epi8(__m256i __S, __mmask32 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_mask_expand_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.mask.expand.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i1> %{{.*}})
  return _mm256_mask_expand_epi8(__S, __U, __D);
}

__m256i test_mm256_maskz_expand_epi8(__mmask32 __U, __m256i __D) {
  // CHECK-LABEL: test_mm256_maskz_expand_epi8
  // CHECK: call <32 x i8> @llvm.x86.avx512.mask.expand.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i1> %{{.*}})
  return _mm256_maskz_expand_epi8(__U, __D);
}

__m256i test_mm256_mask_expandloadu_epi16(__m256i __S, __mmask16 __U, void const* __P) {
  // CHECK-LABEL: test_mm256_mask_expandloadu_epi16
  // CHECK: call <16 x i16> @llvm.masked.expandload.v16i16(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mask_expandloadu_epi16(__S, __U, __P);
}

__m256i test_mm256_maskz_expandloadu_epi16(__mmask16 __U, void const* __P) {
  // CHECK-LABEL: test_mm256_maskz_expandloadu_epi16
  // CHECK: call <16 x i16> @llvm.masked.expandload.v16i16(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_maskz_expandloadu_epi16(__U, __P);
}

__m256i test_mm256_mask_expandloadu_epi8(__m256i __S, __mmask32 __U, void const* __P) {
  // CHECK-LABEL: test_mm256_mask_expandloadu_epi8
  // CHECK: call <32 x  i8> @llvm.masked.expandload.v32i8(ptr %{{.*}}, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_mask_expandloadu_epi8(__S, __U, __P);
}

__m256i test_mm256_maskz_expandloadu_epi8(__mmask32 __U, void const* __P) {
  // CHECK-LABEL: test_mm256_maskz_expandloadu_epi8
  // CHECK: call <32 x i8> @llvm.masked.expandload.v32i8(ptr %{{.*}}, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_maskz_expandloadu_epi8(__U, __P);
}

__m256i test_mm256_mask_shldi_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldi_epi64
  // CHECK: call <4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> splat (i64 47))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_shldi_epi64(__S, __U, __A, __B, 47);
}

__m256i test_mm256_maskz_shldi_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldi_epi64
  // CHECK: call <4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> splat (i64 63))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shldi_epi64(__U, __A, __B, 63);
}

__m256i test_mm256_shldi_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldi_epi64
  // CHECK: call <4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> splat (i64 31))
  return _mm256_shldi_epi64(__A, __B, 31);
}

__m128i test_mm_mask_shldi_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldi_epi64
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 47))
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_shldi_epi64(__S, __U, __A, __B, 47);
}

__m128i test_mm_maskz_shldi_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldi_epi64
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 63))
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_shldi_epi64(__U, __A, __B, 63);
}

__m128i test_mm_shldi_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldi_epi64
  // CHECK: call <2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 31))
  return _mm_shldi_epi64(__A, __B, 31);
}

__m256i test_mm256_mask_shldi_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldi_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> splat (i32 7))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shldi_epi32(__S, __U, __A, __B, 7);
}

__m256i test_mm256_maskz_shldi_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldi_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> splat (i32 15))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shldi_epi32(__U, __A, __B, 15);
}

__m256i test_mm256_shldi_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldi_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> splat (i32 31))
  return _mm256_shldi_epi32(__A, __B, 31);
}

__m128i test_mm_mask_shldi_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldi_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 7))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shldi_epi32(__S, __U, __A, __B, 7);
}

__m128i test_mm_maskz_shldi_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldi_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 15))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shldi_epi32(__U, __A, __B, 15);
}

__m128i test_mm_shldi_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldi_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 31))
  return _mm_shldi_epi32(__A, __B, 31);
}

__m256i test_mm256_mask_shldi_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldi_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> splat (i16 3))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shldi_epi16(__S, __U, __A, __B, 3);
}

__m256i test_mm256_maskz_shldi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldi_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> splat (i16 7))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shldi_epi16(__U, __A, __B, 7);
}

__m256i test_mm256_shldi_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldi_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> splat (i16 31))
  return _mm256_shldi_epi16(__A, __B, 31);
}

__m128i test_mm_mask_shldi_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldi_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 3))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shldi_epi16(__S, __U, __A, __B, 3);
}

__m128i test_mm_maskz_shldi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldi_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 7))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shldi_epi16(__U, __A, __B, 7);
}

__m128i test_mm_shldi_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldi_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 31))
  return _mm_shldi_epi16(__A, __B, 31);
}

__m256i test_mm256_mask_shrdi_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdi_epi64
  // CHECK: call <4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> splat (i64 47))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_shrdi_epi64(__S, __U, __A, __B, 47);
}

__m256i test_mm256_maskz_shrdi_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdi_epi64
  // CHECK: call <4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> splat (i64 63))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shrdi_epi64(__U, __A, __B, 63);
}

__m256i test_mm256_shrdi_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdi_epi64
  // CHECK: call <4 x i64>  @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> splat (i64 31)
  return _mm256_shrdi_epi64(__A, __B, 31);
}

__m128i test_mm_mask_shrdi_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdi_epi64
  // CHECK: call <2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 47))
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_shrdi_epi64(__S, __U, __A, __B, 47);
}

__m128i test_mm_maskz_shrdi_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdi_epi64
  // CHECK: call <2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 63))
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_shrdi_epi64(__U, __A, __B, 63);
}

__m128i test_mm_shrdi_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdi_epi64
  // CHECK: call <2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> splat (i64 31))
  return _mm_shrdi_epi64(__A, __B, 31);
}

__m256i test_mm256_mask_shrdi_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdi_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> splat (i32 7))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shrdi_epi32(__S, __U, __A, __B, 7);
}

__m256i test_mm256_maskz_shrdi_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdi_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> splat (i32 15))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shrdi_epi32(__U, __A, __B, 15);
}

__m256i test_mm256_shrdi_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdi_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> splat (i32 31)
  return _mm256_shrdi_epi32(__A, __B, 31);
}

__m128i test_mm_mask_shrdi_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdi_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 7))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shrdi_epi32(__S, __U, __A, __B, 7);
}

__m128i test_mm_maskz_shrdi_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdi_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 15))
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shrdi_epi32(__U, __A, __B, 15);
}

__m128i test_mm_shrdi_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdi_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 31))
  return _mm_shrdi_epi32(__A, __B, 31);
}

__m256i test_mm256_mask_shrdi_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdi_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> splat (i16 3))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shrdi_epi16(__S, __U, __A, __B, 3);
}

__m256i test_mm256_maskz_shrdi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdi_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> splat (i16 7))
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shrdi_epi16(__U, __A, __B, 7);
}

__m256i test_mm256_shrdi_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdi_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> splat (i16 31))
  return _mm256_shrdi_epi16(__A, __B, 31);
}

__m128i test_mm_mask_shrdi_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdi_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 3))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shrdi_epi16(__S, __U, __A, __B, 3);
}

__m128i test_mm_maskz_shrdi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdi_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 7))
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shrdi_epi16(__U, __A, __B, 7);
}

__m128i test_mm_shrdi_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdi_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> splat (i16 31))
  return _mm_shrdi_epi16(__A, __B, 31);
}

__m256i test_mm256_mask_shldv_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_shldv_epi64(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4di(_mm256_mask_shldv_epi64((__m256i)(__v4di){ -8, 9, 10, -11}, 0x9, (__m256i)(__v4di){ -1, -2, -3, 4}, (__m256i)(__v4di){ -4, -3, 2, 1}), -8070450532247928833LL, 9, 10, -22));

__m256i test_mm256_maskz_shldv_epi64(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shldv_epi64(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4di(_mm256_maskz_shldv_epi64(0x9, (__m256i)(__v4di){ -8, 9, 10, -11}, (__m256i)(__v4di){ -1, -2, -3, 4}, (__m256i)(__v4di){ -4, -3, 2, 1}), -8070450532247928833LL, 0, 0, -22));

__m256i test_mm256_shldv_epi64(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_shldv_epi64(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4di(_mm256_shldv_epi64((__m256i)(__v4di){ -8, 9, 10, -11}, (__m256i)(__v4di){ -1, -2, -3, 4}, (__m256i)(__v4di){ -4, -3, 2, 1}), -8070450532247928833LL, 4611686018427387903LL, 43, -22));

__m128i test_mm_mask_shldv_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_shldv_epi64(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v2di(_mm_mask_shldv_epi64((__m128i)(__v2di){ -4, -5}, 0x1, (__m128i)(__v2di){ -1, 2}, (__m128i)(__v2di){ 2, 1}), -13, -5));

__m128i test_mm_maskz_shldv_epi64(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_shldv_epi64(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v2di(_mm_maskz_shldv_epi64(0x1, (__m128i)(__v2di){ -4, -5}, (__m128i)(__v2di){ -1, 2}, (__m128i)(__v2di){ 2, 1}), -13, 0));

__m128i test_mm_shldv_epi64(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_shldv_epi64(__S, __A, __B);
}
TEST_CONSTEXPR(match_v2di(_mm_shldv_epi64((__m128i)(__v2di){ -4, -5}, (__m128i)(__v2di){ -1, 2}, (__m128i)(__v2di){ 2, 1}), -13, -10));

__m256i test_mm256_mask_shldv_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldv_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shldv_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8si(_mm256_mask_shldv_epi32((__m256i)(__v8si){ 16, -17, -18, -19, -20, -21, -22, 23}, 0xDF, (__m256i)(__v8si){ 1, -2, 3, -4, 5, -6, -7, -8}, (__m256i)(__v8si){ -8, 7, 6, -5, 4, 3, 2, -1}), 268435456, -2049, -1152, 1879048191, -320, -21, -85, -4));

__m256i test_mm256_maskz_shldv_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldv_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shldv_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(_mm256_maskz_shldv_epi32(0xDF, (__m256i)(__v8si){ 16, -17, -18, -19, -20, -21, -22, 23}, (__m256i)(__v8si){ 1, -2, 3, -4, 5, -6, -7, -8}, (__m256i)(__v8si){ -8, 7, 6, -5, 4, 3, 2, -1}), 268435456, -2049, -1152, 1879048191, -320, 0, -85, -4));

__m256i test_mm256_shldv_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldv_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_shldv_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(_mm256_shldv_epi32((__m256i)(__v8si){ 16, -17, -18, -19, -20, -21, -22, 23}, (__m256i)(__v8si){ 1, -2, 3, -4, 5, -6, -7, -8}, (__m256i)(__v8si){ -8, 7, 6, -5, 4, 3, 2, -1}), 268435456, -2049, -1152, 1879048191, -320, -161, -85, -4));

__m128i test_mm_mask_shldv_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldv_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shldv_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(_mm_mask_shldv_epi32((__m128i)(__v4si){ -8, -9, -10, -11}, 0xD, (__m128i)(__v4si){ 1, -2, -3, 4}, (__m128i)(__v4si){ -4, -3, -2, 1}), -2147483648, -9, -1073741825, -22));

__m128i test_mm_maskz_shldv_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldv_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shldv_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(_mm_maskz_shldv_epi32(0xD, (__m128i)(__v4si){ -8, -9, -10, -11}, (__m128i)(__v4si){ 1, -2, -3, 4}, (__m128i)(__v4si){ -4, -3, -2, 1}), -2147483648, 0, -1073741825, -22));

__m128i test_mm_shldv_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldv_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_shldv_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(_mm_shldv_epi32((__m128i)(__v4si){ -8, -9, -10, -11}, (__m128i)(__v4si){ 1, -2, -3, 4}, (__m128i)(__v4si){ -4, -3, -2, 1}), -2147483648, -1, -1073741825, -22));

__m256i test_mm256_mask_shldv_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldv_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shldv_epi16(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shldv_epi16((__m256i)(__v16hi){ 32, -33, 34, 35, -36, 37, -38, -39, -40, 41, -42, 43, -44, 45, -46, 47}, 0x12D6, (__m256i)(__v16hi){ -1, -2, 3, -4, -5, -6, -7, 8, -9, -10, -11, -12, 13, 14, 15, -16}, (__m256i)(__v16hi){ 16, 15, 14, -13, -12, -11, 10, 9, -8, -7, 6, -5, -4, 3, -2, 1}), 32, -1, -32768, 35, -561, 37, 27647, -19968, -40, 21503, -42, 43, 16384, 45, -46, 47));

__m256i test_mm256_maskz_shldv_epi16(__mmask16 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldv_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shldv_epi16(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shldv_epi16(0x12D6, (__m256i)(__v16hi){ 32, -33, 34, 35, -36, 37, -38, -39, -40, 41, -42, 43, -44, 45, -46, 47}, (__m256i)(__v16hi){ -1, -2, 3, -4, -5, -6, -7, 8, -9, -10, -11, -12, 13, 14, 15, -16}, (__m256i)(__v16hi){ 16, 15, 14, -13, -12, -11, 10, 9, -8, -7, 6, -5, -4, 3, -2, 1}), 0, -1, -32768, 0, -561, 0, 27647, -19968, 0, 21503, 0, 0, 16384, 0, 0, 0));

__m256i test_mm256_shldv_epi16(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldv_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_shldv_epi16(__S, __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_shldv_epi16((__m256i)(__v16hi){ 32, -33, 34, 35, -36, 37, -38, -39, -40, 41, -42, 43, -44, 45, -46, 47}, (__m256i)(__v16hi){ -1, -2, 3, -4, -5, -6, -7, 8, -9, -10, -11, -12, 13, 14, 15, -16}, (__m256i)(__v16hi){ 16, 15, 14, -13, -12, -11, 10, 9, -8, -7, 6, -5, -4, 3, -2, 1}), 32, -1, -32768, 287, -561, 1215, 27647, -19968, -9985, 21503, -2625, 24575, 16384, 360, -32765, 95));

__m128i test_mm_mask_shldv_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldv_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shldv_epi16(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_shldv_epi16((__m128i)(__v8hi){ -16, 17, -18, -19, 20, 21, 22, -23}, 0x3A, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, 6, -7, -8}, (__m128i)(__v8hi){ 8, -7, -6, 5, 4, 3, 2, -1}), -16, 8704, -18, -577, 335, 168, 22, -23));

__m128i test_mm_maskz_shldv_epi16(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldv_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shldv_epi16(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shldv_epi16(0x3A, (__m128i)(__v8hi){ -16, 17, -18, -19, 20, 21, 22, -23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, 6, -7, -8}, (__m128i)(__v8hi){ 8, -7, -6, 5, 4, 3, 2, -1}), 0, 8704, 0, -577, 335, 168, 0, 0));

__m128i test_mm_shldv_epi16(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldv_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_shldv_epi16(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_shldv_epi16((__m128i)(__v8hi){ -16, 17, -18, -19, 20, 21, 22, -23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, 6, -7, -8}, (__m128i)(__v8hi){ 8, -7, -6, 5, 4, 3, 2, -1}), -4096, 8704, -18432, -577, 335, 168, 91, -4));

__m256i test_mm256_mask_shrdv_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_shrdv_epi64(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4di(_mm256_mask_shrdv_epi64((__m256i)(__v4di){ -8, 9, 10, -11}, 0x9, (__m256i)(__v4di){ -1, -2, -3, 4}, (__m256i)(__v4di){ -4, -3, 2, 1}), -1, 9, 10, 9223372036854775802LL));

__m256i test_mm256_maskz_shrdv_epi64(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shrdv_epi64(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4di(_mm256_maskz_shrdv_epi64(0x9, (__m256i)(__v4di){ -8, 9, 10, -11}, (__m256i)(__v4di){ -1, -2, -3, 4}, (__m256i)(__v4di){ -4, -3, 2, 1}), -1, 0, 0, 9223372036854775802LL));

__m256i test_mm256_shrdv_epi64(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_shrdv_epi64(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4di(_mm256_shrdv_epi64((__m256i)(__v4di){ -8, 9, 10, -11}, (__m256i)(__v4di){ -1, -2, -3, 4}, (__m256i)(__v4di){ -4, -3, 2, 1}), -1, -16, 4611686018427387906LL, 9223372036854775802LL));

__m128i test_mm_mask_shrdv_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_shrdv_epi64(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v2di(_mm_mask_shrdv_epi64((__m128i)(__v2di){ -4, -5}, 0x1, (__m128i)(__v2di){ -1, 2}, (__m128i)(__v2di){ 2, 1}), -1, -5));

__m128i test_mm_maskz_shrdv_epi64(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_shrdv_epi64(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v2di(_mm_maskz_shrdv_epi64(0x1, (__m128i)(__v2di){ -4, -5}, (__m128i)(__v2di){ -1, 2}, (__m128i)(__v2di){ 2, 1}), -1, 0));

__m128i test_mm_shrdv_epi64(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_shrdv_epi64(__S, __A, __B);
}
TEST_CONSTEXPR(match_v2di(_mm_shrdv_epi64((__m128i)(__v2di){ -4, -5}, (__m128i)(__v2di){ -1, 2}, (__m128i)(__v2di){ 2, 1}), -1, 9223372036854775805LL));

__m256i test_mm256_mask_shrdv_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdv_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shrdv_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8si(_mm256_mask_shrdv_epi32((__m256i)(__v8si){ 16, -17, -18, -19, -20, -21, -22, 23}, 0xDF, (__m256i)(__v8si){ 1, -2, 3, -4, 5, -6, -7, -8}, (__m256i)(__v8si){ -8, 7, 6, -5, 4, 3, 2, -1}), 256, -33554433, 268435455, -97, 1610612734, -21, 2147483642, -16));

__m256i test_mm256_maskz_shrdv_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdv_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shrdv_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(_mm256_maskz_shrdv_epi32(0xDF, (__m256i)(__v8si){ 16, -17, -18, -19, -20, -21, -22, 23}, (__m256i)(__v8si){ 1, -2, 3, -4, 5, -6, -7, -8}, (__m256i)(__v8si){ -8, 7, 6, -5, 4, 3, 2, -1}), 256, -33554433, 268435455, -97, 1610612734, 0, 2147483642, -16));

__m256i test_mm256_shrdv_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdv_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_shrdv_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8si(_mm256_shrdv_epi32((__m256i)(__v8si){ 16, -17, -18, -19, -20, -21, -22, 23}, (__m256i)(__v8si){ 1, -2, 3, -4, 5, -6, -7, -8}, (__m256i)(__v8si){ -8, 7, 6, -5, 4, 3, 2, -1}), 256, -33554433, 268435455, -97, 1610612734, 1610612733, 2147483642, -16));

__m128i test_mm_mask_shrdv_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdv_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shrdv_epi32(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v4si(_mm_mask_shrdv_epi32((__m128i)(__v4si){ -8, -9, -10, -11}, 0xD, (__m128i)(__v4si){ 1, -2, -3, 4}, (__m128i)(__v4si){ -4, -3, -2, 1}), 31, -9, -9, 2147483642));

__m128i test_mm_maskz_shrdv_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdv_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shrdv_epi32(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(_mm_maskz_shrdv_epi32(0xD, (__m128i)(__v4si){ -8, -9, -10, -11}, (__m128i)(__v4si){ 1, -2, -3, 4}, (__m128i)(__v4si){ -4, -3, -2, 1}), 31, 0, -9, 2147483642));

__m128i test_mm_shrdv_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdv_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_shrdv_epi32(__S, __A, __B);
}
TEST_CONSTEXPR(match_v4si(_mm_shrdv_epi32((__m128i)(__v4si){ -8, -9, -10, -11}, (__m128i)(__v4si){ 1, -2, -3, 4}, (__m128i)(__v4si){ -4, -3, -2, 1}), 31, -9, -9, 2147483642));

__m256i test_mm256_mask_shrdv_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdv_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shrdv_epi16(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shrdv_epi16((__m256i)(__v16hi){ 32, -33, 34, 35, -36, 37, -38, -39, -40, 41, -42, 43, -44, 45, -46, 47}, 0x12D6, (__m256i)(__v16hi){ -1, -2, 3, -4, -5, -6, -7, 8, -9, -10, -11, -12, 13, 14, 15, -16}, (__m256i)(__v16hi){ 16, 15, 14, -13, -12, -11, 10, 9, -8, -7, 6, -5, -4, 3, -2, 1}), 32, -3, 12, 35, -16387, 37, -385, 1151, -40, -1280, -42, 43, 223, 45, -46, 47));

__m256i test_mm256_maskz_shrdv_epi16(__mmask16 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdv_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shrdv_epi16(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shrdv_epi16(0x12D6, (__m256i)(__v16hi){ 32, -33, 34, 35, -36, 37, -38, -39, -40, 41, -42, 43, -44, 45, -46, 47}, (__m256i)(__v16hi){ -1, -2, 3, -4, -5, -6, -7, 8, -9, -10, -11, -12, 13, 14, 15, -16}, (__m256i)(__v16hi){ 16, 15, 14, -13, -12, -11, 10, 9, -8, -7, 6, -5, -4, 3, -2, 1}), 0, -3, 12, 0, -16387, 0, -385, 1151, 0, -1280, 0, 0, 223, 0, 0, 0));

__m256i test_mm256_shrdv_epi16(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdv_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_shrdv_epi16(__S, __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_shrdv_epi16((__m256i)(__v16hi){ 32, -33, 34, 35, -36, 37, -38, -39, -40, 41, -42, 43, -44, 45, -46, 47}, (__m256i)(__v16hi){ -1, -2, 3, -4, -5, -6, -7, 8, -9, -10, -11, -12, 13, 14, 15, -16}, (__m256i)(__v16hi){ 16, 15, 14, -13, -12, -11, 10, 9, -8, -7, 6, -5, -4, 3, -2, 1}), 32, -3, 12, -32764, -16387, -12287, -385, 1151, -2049, -1280, -10241, -384, 223, -16379, 63, 23));

__m128i test_mm_mask_shrdv_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdv_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shrdv_epi16(__S, __U, __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_shrdv_epi16((__m128i)(__v8hi){ -16, 17, -18, -19, 20, 21, 22, -23}, 0x3A, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, 6, -7, -8}, (__m128i)(__v8hi){ 8, -7, -6, 5, 4, 3, 2, -1}), -16, 256, -18, -6145, -20479, -16382, 22, -23));

__m128i test_mm_maskz_shrdv_epi16(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdv_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shrdv_epi16(__U, __S, __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shrdv_epi16(0x3A, (__m128i)(__v8hi){ -16, 17, -18, -19, 20, 21, 22, -23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, 6, -7, -8}, (__m128i)(__v8hi){ 8, -7, -6, 5, 4, 3, 2, -1}), 0, 256, 0, -6145, -20479, -16382, 0, 0));

__m128i test_mm_shrdv_epi16(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdv_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_shrdv_epi16(__S, __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_shrdv_epi16((__m128i)(__v8hi){ -16, 17, -18, -19, 20, 21, 22, -23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, 6, -7, -8}, (__m128i)(__v8hi){ 8, -7, -6, 5, 4, 3, 2, -1}), 511, 256, 255, -6145, -20479, -16382, 16389, -15));

