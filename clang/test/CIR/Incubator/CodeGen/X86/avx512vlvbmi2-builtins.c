// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512vbmi2 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512vl -target-feature +avx512vbmi2 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

__m128i test_mm_mask_expandloadu_epi16(__m128i __S, __mmask8 __U, void const* __P) {
  // CIR-LABEL: _mm_mask_expandloadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM-LABEL: @test_mm_mask_expandloadu_epi16
  // LLVM: @llvm.masked.expandload.v8i16(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mask_expandloadu_epi16(__S, __U, __P);
}

__m128i test_mm_maskz_expandloadu_epi16(__mmask8 __U, void const* __P) {
  // CIR-LABEL: _mm_maskz_expandloadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 8>>, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM-LABEL: @test_mm_maskz_expandloadu_epi16
  // LLVM: @llvm.masked.expandload.v8i16(ptr %{{.*}}, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_maskz_expandloadu_epi16(__U, __P);
}

__m256i test_mm256_mask_expandloadu_epi16(__m256i __S, __mmask16 __U, void const* __P) {
  // CIR-LABEL: _mm256_mask_expandloadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s16i x 16>) -> !cir.vector<!s16i x 16>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_epi16
  // LLVM: @llvm.masked.expandload.v16i16(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mask_expandloadu_epi16(__S, __U, __P);
}

__m256i test_mm256_maskz_expandloadu_epi16(__mmask16 __U, void const* __P) {
  // CIR-LABEL: _mm256_maskz_expandloadu_epi16
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s16i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s16i x 16>) -> !cir.vector<!s16i x 16>

  // LLVM-LABEL: @test_mm256_maskz_expandloadu_epi16
  // LLVM: @llvm.masked.expandload.v16i16(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
return _mm256_maskz_expandloadu_epi16(__U, __P);
}

__m128i test_mm_mask_expandloadu_epi8(__m128i __S, __mmask16 __U, void const* __P) {
   // CIR-LABEL: _mm_mask_expandloadu_epi8
   // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s8i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

   // LLVM-LABEL: @test_mm_mask_expandloadu_epi8
   // LLVM: @llvm.masked.expandload.v16i8(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
   return _mm_mask_expandloadu_epi8(__S, __U, __P);
}

__m128i test_mm_maskz_expandloadu_epi8(__mmask16 __U, void const* __P) {
   // CIR-LABEL: _mm_maskz_expandloadu_epi8
   // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s8i x 16>>, !cir.vector<!cir.int<s, 1> x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

   // LLVM-LABEL: @test_mm_maskz_expandloadu_epi8
   // LLVM: @llvm.masked.expandload.v16i8(ptr %{{.*}}, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
return _mm_maskz_expandloadu_epi8(__U, __P);
}

__m256i test_mm256_mask_expandloadu_epi8(__m256i __S, __mmask32 __U, void const* __P) {
  // CIR-LABEL: _mm256_mask_expandloadu_epi8
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s8i x 32>>, !cir.vector<!cir.int<s, 1> x 32>, !cir.vector<!s8i x 32>) -> !cir.vector<!s8i x 32>

  // LLVM-LABEL: @test_mm256_mask_expandloadu_epi8
  // LLVM: @llvm.masked.expandload.v32i8(ptr %{{.*}}, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_mask_expandloadu_epi8(__S, __U, __P);
}

__m256i test_mm256_maskz_expandloadu_epi8(__mmask32 __U, void const* __P) {
   // CIR-LABEL: _mm256_maskz_expandloadu_epi8
   // CIR: %{{.*}} = cir.llvm.intrinsic "masked.expandload" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!s8i x 32>>, !cir.vector<!cir.int<s, 1> x 32>, !cir.vector<!s8i x 32>) -> !cir.vector<!s8i x 32>

   // LLVM-LABEL: @test_mm256_maskz_expandloadu_epi8
   // LLVM: @llvm.masked.expandload.v32i8(ptr %{{.*}}, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
   return _mm256_maskz_expandloadu_epi8(__U, __P);
}

void test_mm256_mask_compressstoreu_epi16(void *__P, __mmask16 __U, __m256i __D) {
  // CIR-LABEL: _mm256_mask_compressstoreu_epi16
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s16i x 16>, !cir.ptr<!cir.vector<!s16i x 16>>, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: @test_mm256_mask_compressstoreu_epi16
  // LLVM: @llvm.masked.compressstore.v16i16(<16 x i16> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  _mm256_mask_compressstoreu_epi16(__P, __U, __D);
}

void test_mm_mask_compressstoreu_epi8(void *__P, __mmask16 __U, __m128i __D) {
  // CIR-LABEL: _mm_mask_compressstoreu_epi8
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s8i x 16>, !cir.ptr<!cir.vector<!s8i x 16>>, !cir.vector<!cir.int<s, 1> x 16>) -> !void

  // LLVM-LABEL: @test_mm_mask_compressstoreu_epi8
  // LLVM: @llvm.masked.compressstore.v16i8(<16 x i8> %{{.*}}, ptr %{{.*}}, <16 x i1> %{{.*}})
  _mm_mask_compressstoreu_epi8(__P, __U, __D);
}

void test_mm256_mask_compressstoreu_epi8(void *__P, __mmask32 __U, __m256i __D) {
  // CIR-LABEL: _mm256_mask_compressstoreu_epi8
  // CIR: cir.llvm.intrinsic "masked.compressstore" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!s8i x 32>, !cir.ptr<!cir.vector<!s8i x 32>>, !cir.vector<!cir.int<s, 1> x 32>) -> !void

  // LLVM-LABEL: @test_mm256_mask_compressstoreu_epi8
  // LLVM: @llvm.masked.compressstore.v32i8(<32 x i8> %{{.*}}, ptr %{{.*}}, <32 x i1> %{{.*}})
  _mm256_mask_compressstoreu_epi8(__P, __U, __D);
}