// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512vbmi2 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

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

__m256i test_mm256_maskz_shldv_epi64(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shldv_epi64(__U, __S, __A, __B);
}

__m256i test_mm256_shldv_epi64(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshl.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_shldv_epi64(__S, __A, __B);
}

__m128i test_mm_mask_shldv_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_shldv_epi64(__S, __U, __A, __B);
}

__m128i test_mm_maskz_shldv_epi64(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_shldv_epi64(__U, __S, __A, __B);
}

__m128i test_mm_shldv_epi64(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshl.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_shldv_epi64(__S, __A, __B);
}

__m256i test_mm256_mask_shldv_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldv_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shldv_epi32(__S, __U, __A, __B);
}

__m256i test_mm256_maskz_shldv_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldv_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shldv_epi32(__U, __S, __A, __B);
}

__m256i test_mm256_shldv_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldv_epi32
  // CHECK: call <8 x i32> @llvm.fshl.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_shldv_epi32(__S, __A, __B);
}

__m128i test_mm_mask_shldv_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldv_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shldv_epi32(__S, __U, __A, __B);
}

__m128i test_mm_maskz_shldv_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldv_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shldv_epi32(__U, __S, __A, __B);
}

__m128i test_mm_shldv_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldv_epi32
  // CHECK: call <4 x i32> @llvm.fshl.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_shldv_epi32(__S, __A, __B);
}

__m256i test_mm256_mask_shldv_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shldv_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shldv_epi16(__S, __U, __A, __B);
}

__m256i test_mm256_maskz_shldv_epi16(__mmask16 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shldv_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shldv_epi16(__U, __S, __A, __B);
}

__m256i test_mm256_shldv_epi16(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shldv_epi16
  // CHECK: call <16 x i16> @llvm.fshl.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_shldv_epi16(__S, __A, __B);
}

__m128i test_mm_mask_shldv_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shldv_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shldv_epi16(__S, __U, __A, __B);
}

__m128i test_mm_maskz_shldv_epi16(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shldv_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shldv_epi16(__U, __S, __A, __B);
}

__m128i test_mm_shldv_epi16(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shldv_epi16
  // CHECK: call <8 x i16> @llvm.fshl.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_shldv_epi16(__S, __A, __B);
}

__m256i test_mm256_mask_shrdv_epi64(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_shrdv_epi64(__S, __U, __A, __B);
}

__m256i test_mm256_maskz_shrdv_epi64(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_shrdv_epi64(__U, __S, __A, __B);
}

__m256i test_mm256_shrdv_epi64(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdv_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.fshr.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_shrdv_epi64(__S, __A, __B);
}

__m128i test_mm_mask_shrdv_epi64(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_shrdv_epi64(__S, __U, __A, __B);
}

__m128i test_mm_maskz_shrdv_epi64(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_shrdv_epi64(__U, __S, __A, __B);
}

__m128i test_mm_shrdv_epi64(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdv_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.fshr.v2i64(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_shrdv_epi64(__S, __A, __B);
}

__m256i test_mm256_mask_shrdv_epi32(__m256i __S, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdv_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shrdv_epi32(__S, __U, __A, __B);
}

__m256i test_mm256_maskz_shrdv_epi32(__mmask8 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdv_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shrdv_epi32(__U, __S, __A, __B);
}

__m256i test_mm256_shrdv_epi32(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdv_epi32
  // CHECK: call <8 x i32> @llvm.fshr.v8i32(<8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_shrdv_epi32(__S, __A, __B);
}

__m128i test_mm_mask_shrdv_epi32(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdv_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shrdv_epi32(__S, __U, __A, __B);
}

__m128i test_mm_maskz_shrdv_epi32(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdv_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shrdv_epi32(__U, __S, __A, __B);
}

__m128i test_mm_shrdv_epi32(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdv_epi32
  // CHECK: call <4 x i32> @llvm.fshr.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_shrdv_epi32(__S, __A, __B);
}

__m256i test_mm256_mask_shrdv_epi16(__m256i __S, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shrdv_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shrdv_epi16(__S, __U, __A, __B);
}

__m256i test_mm256_maskz_shrdv_epi16(__mmask16 __U, __m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shrdv_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shrdv_epi16(__U, __S, __A, __B);
}

__m256i test_mm256_shrdv_epi16(__m256i __S, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_shrdv_epi16
  // CHECK: call <16 x i16> @llvm.fshr.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_shrdv_epi16(__S, __A, __B);
}

__m128i test_mm_mask_shrdv_epi16(__m128i __S, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shrdv_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shrdv_epi16(__S, __U, __A, __B);
}

__m128i test_mm_maskz_shrdv_epi16(__mmask8 __U, __m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shrdv_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shrdv_epi16(__U, __S, __A, __B);
}

__m128i test_mm_shrdv_epi16(__m128i __S, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_shrdv_epi16
  // CHECK: call <8 x i16> @llvm.fshr.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_shrdv_epi16(__S, __A, __B);
}

