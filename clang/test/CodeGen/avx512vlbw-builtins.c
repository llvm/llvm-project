// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__mmask32 test_mm256_cmpeq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.256
  return (__mmask32)_mm256_cmpeq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.256
  return (__mmask32)_mm256_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.128
  return (__mmask16)_mm_cmpeq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.128
  return (__mmask16)_mm_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.256
  return (__mmask16)_mm256_cmpeq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.256
  return (__mmask16)_mm256_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.128
  return (__mmask8)_mm_cmpeq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.128
  return (__mmask8)_mm_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.256
  return (__mmask32)_mm256_cmpgt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.256
  return (__mmask32)_mm256_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.128
  return (__mmask16)_mm_cmpgt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.b.128
  return (__mmask16)_mm_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.256
  return (__mmask16)_mm256_cmpgt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.256
  return (__mmask16)_mm256_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.128
  return (__mmask8)_mm_cmpgt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.w.128
  return (__mmask8)_mm_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 0, i16 -1)
  return (__mmask64)_mm_cmpeq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 0, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 0, i8 -1)
  return (__mmask32)_mm_cmpeq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 0, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpeq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 0, i32 -1)
  return (__mmask64)_mm256_cmpeq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 0, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 0, i16 -1)
  return (__mmask32)_mm256_cmpeq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 0, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 6, i16 -1)
  return (__mmask64)_mm_cmpgt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 6, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 6, i8 -1)
  return (__mmask32)_mm_cmpgt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 6, i32 -1)
  return (__mmask64)_mm256_cmpgt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 6, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 6, i16 -1)
  return (__mmask32)_mm256_cmpgt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 6, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 5, i16 -1)
  return (__mmask64)_mm_cmpge_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 5, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 5, i16 -1)
  return (__mmask64)_mm_cmpge_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 5, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 5, i8 -1)
  return (__mmask32)_mm_cmpge_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 5, i8 -1)
  return (__mmask32)_mm_cmpge_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 5, i32 -1)
  return (__mmask64)_mm256_cmpge_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 5, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 5, i32 -1)
  return (__mmask64)_mm256_cmpge_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 5, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 5, i16 -1)
  return (__mmask32)_mm256_cmpge_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 5, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 5, i16 -1)
  return (__mmask32)_mm256_cmpge_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 5, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 2, i16 -1)
  return (__mmask64)_mm_cmple_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 2, i16 {{.*}})
  return (__mmask64)_mm_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 2, i16 -1)
  return (__mmask64)_mm_cmple_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 2, i16 {{.*}})
  return (__mmask64)_mm_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 2, i8 -1)
  return (__mmask32)_mm_cmple_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask32)_mm_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 2, i8 -1)
  return (__mmask32)_mm_cmple_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask32)_mm_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 2, i32 -1)
  return (__mmask64)_mm256_cmple_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 2, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 2, i32 -1)
  return (__mmask64)_mm256_cmple_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 2, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 2, i16 -1)
  return (__mmask32)_mm256_cmple_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 2, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 2, i16 -1)
  return (__mmask32)_mm256_cmple_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 2, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 1, i16 -1)
  return (__mmask64)_mm_cmplt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 1, i16 {{.*}})
  return (__mmask64)_mm_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 1, i16 -1)
  return (__mmask64)_mm_cmplt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 1, i16 {{.*}})
  return (__mmask64)_mm_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 1, i8 -1)
  return (__mmask32)_mm_cmplt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask32)_mm_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 1, i8 -1)
  return (__mmask32)_mm_cmplt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask32)_mm_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 1, i32 -1)
  return (__mmask64)_mm256_cmplt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 1, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 1, i32 -1)
  return (__mmask64)_mm256_cmplt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 1, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 1, i16 -1)
  return (__mmask32)_mm256_cmplt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 1, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 1, i16 -1)
  return (__mmask32)_mm256_cmplt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 1, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 4, i16 -1)
  return (__mmask64)_mm_cmpneq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 4, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 4, i16 -1)
  return (__mmask64)_mm_cmpneq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 4, i16 {{.*}})
  return (__mmask64)_mm_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 4, i8 -1)
  return (__mmask32)_mm_cmpneq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 4, i8 -1)
  return (__mmask32)_mm_cmpneq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask32)_mm_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 4, i32 -1)
  return (__mmask64)_mm256_cmpneq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 4, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 4, i32 -1)
  return (__mmask64)_mm256_cmpneq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 4, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 4, i16 -1)
  return (__mmask32)_mm256_cmpneq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 4, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 4, i16 -1)
  return (__mmask32)_mm256_cmpneq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 4, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmp_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 7, i16 -1)
  return (__mmask64)_mm_cmp_epi8_mask(__a, __b, 7);
}

__mmask16 test_mm_mask_cmp_epi8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 7, i16 {{.*}})
  return (__mmask64)_mm_mask_cmp_epi8_mask(__u, __a, __b, 7);
}

__mmask16 test_mm_cmp_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 7, i16 -1)
  return (__mmask64)_mm_cmp_epu8_mask(__a, __b, 7);
}

__mmask16 test_mm_mask_cmp_epu8_mask(__mmask64 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.128(<16 x i8> {{.*}}, <16 x i8> {{.*}}, i32 7, i16 {{.*}})
  return (__mmask64)_mm_mask_cmp_epu8_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 7, i8 -1)
  return (__mmask32)_mm_cmp_epi16_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask32)_mm_mask_cmp_epi16_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 7, i8 -1)
  return (__mmask32)_mm_cmp_epu16_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu16_mask(__mmask32 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.128(<8 x i16> {{.*}}, <8 x i16> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask32)_mm_mask_cmp_epu16_mask(__u, __a, __b, 7);
}

__mmask32 test_mm256_cmp_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 7, i32 -1)
  return (__mmask64)_mm256_cmp_epi8_mask(__a, __b, 7);
}

__mmask32 test_mm256_mask_cmp_epi8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 7, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmp_epi8_mask(__u, __a, __b, 7);
}

__mmask32 test_mm256_cmp_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 7, i32 -1)
  return (__mmask64)_mm256_cmp_epu8_mask(__a, __b, 7);
}

__mmask32 test_mm256_mask_cmp_epu8_mask(__mmask64 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu8_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.b.256(<32 x i8> {{.*}}, <32 x i8> {{.*}}, i32 7, i32 {{.*}})
  return (__mmask64)_mm256_mask_cmp_epu8_mask(__u, __a, __b, 7);
}

__mmask16 test_mm256_cmp_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 7, i16 -1)
  return (__mmask32)_mm256_cmp_epi16_mask(__a, __b, 7);
}

__mmask16 test_mm256_mask_cmp_epi16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 7, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmp_epi16_mask(__u, __a, __b, 7);
}

__mmask16 test_mm256_cmp_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 7, i16 -1)
  return (__mmask32)_mm256_cmp_epu16_mask(__a, __b, 7);
}

__mmask16 test_mm256_mask_cmp_epu16_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu16_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.w.256(<16 x i16> {{.*}}, <16 x i16> {{.*}}, i32 7, i16 {{.*}})
  return (__mmask32)_mm256_mask_cmp_epu16_mask(__u, __a, __b, 7);
}


__m256i test_mm256_mask_add_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B){
  //CHECK-LABEL: @test_mm256_mask_add_epi8
  //CHECK: @llvm.x86.avx512.mask.padd.b.256
  return _mm256_mask_add_epi8(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_add_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi8
  //CHECK: @llvm.x86.avx512.mask.padd.b.256
  return _mm256_maskz_add_epi8(__U , __A, __B);
}
__m256i test_mm256_mask_add_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi16
  //CHECK: @llvm.x86.avx512.mask.padd.w.256
  return _mm256_mask_add_epi16(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_add_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi16
  //CHECK: @llvm.x86.avx512.mask.padd.w.256
  return _mm256_maskz_add_epi16(__U , __A, __B);
}

__m256i test_mm256_mask_sub_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi8
  //CHECK: @llvm.x86.avx512.mask.psub.b.256
  return _mm256_mask_sub_epi8(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_sub_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi8
  //CHECK: @llvm.x86.avx512.mask.psub.b.256
  return _mm256_maskz_sub_epi8(__U , __A, __B);
}

__m256i test_mm256_mask_sub_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi16
  //CHECK: @llvm.x86.avx512.mask.psub.w.256
  return _mm256_mask_sub_epi16(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_sub_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi16
  //CHECK: @llvm.x86.avx512.mask.psub.w.256
  return _mm256_maskz_sub_epi16(__U , __A, __B);
}
__m128i test_mm_mask_add_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi8
  //CHECK: @llvm.x86.avx512.mask.padd.b.128
  return _mm_mask_add_epi8(__W, __U , __A, __B);
}

__m128i test_mm_maskz_add_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi8
  //CHECK: @llvm.x86.avx512.mask.padd.b.128
  return _mm_maskz_add_epi8(__U , __A, __B);
}

__m128i test_mm_mask_add_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi16
  //CHECK: @llvm.x86.avx512.mask.padd.w.128
  return _mm_mask_add_epi16(__W, __U , __A, __B);
}

__m128i test_mm_maskz_add_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi16
  //CHECK: @llvm.x86.avx512.mask.padd.w.128
  return _mm_maskz_add_epi16(__U , __A, __B);
}

__m128i test_mm_mask_sub_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi8
  //CHECK: @llvm.x86.avx512.mask.psub.b.128
  return _mm_mask_sub_epi8(__W, __U , __A, __B);
}

__m128i test_mm_maskz_sub_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi8
  //CHECK: @llvm.x86.avx512.mask.psub.b.128
  return _mm_maskz_sub_epi8(__U , __A, __B);
}

__m128i test_mm_mask_sub_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi16
  //CHECK: @llvm.x86.avx512.mask.psub.w.128
  return _mm_mask_sub_epi16(__W, __U , __A, __B);
}

__m128i test_mm_maskz_sub_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi16
  //CHECK: @llvm.x86.avx512.mask.psub.w.128
  return _mm_maskz_sub_epi16(__U , __A, __B);
}

__m256i test_mm256_mask_mullo_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_mullo_epi16
  //CHECK: @llvm.x86.avx512.mask.pmull.w.256
  return _mm256_mask_mullo_epi16(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_mullo_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_mullo_epi16
  //CHECK: @llvm.x86.avx512.mask.pmull.w.256
  return _mm256_maskz_mullo_epi16(__U , __A, __B);
}

__m128i test_mm_mask_mullo_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_mullo_epi16
  //CHECK: @llvm.x86.avx512.mask.pmull.w.128
  return _mm_mask_mullo_epi16(__W, __U , __A, __B);
}

__m128i test_mm_maskz_mullo_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_mullo_epi16
  //CHECK: @llvm.x86.avx512.mask.pmull.w.128
  return _mm_maskz_mullo_epi16(__U , __A, __B);
}


__m128i test_mm_mask_blend_epi8(__mmask16 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi8
  // CHECK: @llvm.x86.avx512.mask.blend.b.128
  return _mm_mask_blend_epi8(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi8(__mmask32 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi8
  // CHECK: @llvm.x86.avx512.mask.blend.b.256
  return _mm256_mask_blend_epi8(__U,__A,__W); 
}

__m128i test_mm_mask_blend_epi16(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi16
  // CHECK: @llvm.x86.avx512.mask.blend.w.128
  return _mm_mask_blend_epi16(__U,__A,__W); 
}

__m256i test_mm256_mask_blend_epi16(__mmask16 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi16
  // CHECK: @llvm.x86.avx512.mask.blend.w.256
  return _mm256_mask_blend_epi16(__U,__A,__W); 
}

__m128i test_mm_mask_abs_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.128
  return _mm_mask_abs_epi8(__W,__U,__A); 
}

__m128i test_mm_maskz_abs_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.128
  return _mm_maskz_abs_epi8(__U,__A); 
}

__m256i test_mm256_mask_abs_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.256
  return _mm256_mask_abs_epi8(__W,__U,__A); 
}

__m256i test_mm256_maskz_abs_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi8
  // CHECK: @llvm.x86.avx512.mask.pabs.b.256
  return _mm256_maskz_abs_epi8(__U,__A); 
}

__m128i test_mm_mask_abs_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.128
  return _mm_mask_abs_epi16(__W,__U,__A); 
}

__m128i test_mm_maskz_abs_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.128
  return _mm_maskz_abs_epi16(__U,__A); 
}

__m256i test_mm256_mask_abs_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.256
  return _mm256_mask_abs_epi16(__W,__U,__A); 
}

__m256i test_mm256_maskz_abs_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi16
  // CHECK: @llvm.x86.avx512.mask.pabs.w.256
  return _mm256_maskz_abs_epi16(__U,__A); 
}

__m128i test_mm_maskz_packs_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.128
  return _mm_maskz_packs_epi32(__M,__A,__B); 
}
__m128i test_mm_mask_packs_epi32(__m128i __W, __mmask16 __M, __m128i __A,          __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.128
  return _mm_mask_packs_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_packs_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.256
  return _mm256_maskz_packs_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_packs_epi32(__m256i __W, __mmask16 __M, __m256i __A,       __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packs_epi32
  // CHECK: @llvm.x86.avx512.mask.packssdw.256
  return _mm256_mask_packs_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_packs_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.128
  return _mm_maskz_packs_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_packs_epi16(__m128i __W, __mmask16 __M, __m128i __A,          __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.128
  return _mm_mask_packs_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_packs_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.256
  return _mm256_maskz_packs_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_packs_epi16(__m256i __W, __mmask32 __M, __m256i __A,       __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packs_epi16
  // CHECK: @llvm.x86.avx512.mask.packsswb.256
  return _mm256_mask_packs_epi16(__W,__M,__A,__B); 
}

__m128i test_mm_mask_packus_epi32(__m128i __W, __mmask16 __M, __m128i __A,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.128
  return _mm_mask_packus_epi32(__W,__M,__A,__B); 
}

__m128i test_mm_maskz_packus_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.128
  return _mm_maskz_packus_epi32(__M,__A,__B); 
}

__m256i test_mm256_maskz_packus_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.256
  return _mm256_maskz_packus_epi32(__M,__A,__B); 
}

__m256i test_mm256_mask_packus_epi32(__m256i __W, __mmask16 __M, __m256i __A,        __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packus_epi32
  // CHECK: @llvm.x86.avx512.mask.packusdw.256
  return _mm256_mask_packus_epi32(__W,__M,__A,__B); 
}

__m128i test_mm_maskz_packus_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.128
  return _mm_maskz_packus_epi16(__M,__A,__B); 
}

__m128i test_mm_mask_packus_epi16(__m128i __W, __mmask16 __M, __m128i __A,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.128
  return _mm_mask_packus_epi16(__W,__M,__A,__B); 
}

__m256i test_mm256_maskz_packus_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.256
  return _mm256_maskz_packus_epi16(__M,__A,__B); 
}

__m256i test_mm256_mask_packus_epi16(__m256i __W, __mmask32 __M, __m256i __A,        __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packus_epi16
  // CHECK: @llvm.x86.avx512.mask.packuswb.256
  return _mm256_mask_packus_epi16(__W,__M,__A,__B); 
}

__m128i test_mm_mask_adds_epi8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.128
  return _mm_mask_adds_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.128
  return _mm_maskz_adds_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epi8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.256
  return _mm256_mask_adds_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epi8
  // CHECK: @llvm.x86.avx512.mask.padds.b.256
  return _mm256_maskz_adds_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epi16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.128
  return _mm_mask_adds_epi16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.128
  return _mm_maskz_adds_epi16(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epi16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.256
  return _mm256_mask_adds_epi16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epi16
  // CHECK: @llvm.x86.avx512.mask.padds.w.256
  return _mm256_maskz_adds_epi16(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epu8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.128
  return _mm_mask_adds_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.128
  return _mm_maskz_adds_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epu8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.256
  return _mm256_mask_adds_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epu8
  // CHECK: @llvm.x86.avx512.mask.paddus.b.256
  return _mm256_maskz_adds_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epu16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.128
  return _mm_mask_adds_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.128
  return _mm_maskz_adds_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epu16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.256
  return _mm256_mask_adds_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epu16
  // CHECK: @llvm.x86.avx512.mask.paddus.w.256
  return _mm256_maskz_adds_epu16(__U,__A,__B); 
}
__m128i test_mm_mask_avg_epu8(__m128i __W, __mmask16 __U, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.128
  return _mm_mask_avg_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_avg_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.128
  return _mm_maskz_avg_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_avg_epu8(__m256i __W, __mmask32 __U, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.256
  return _mm256_mask_avg_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_avg_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_avg_epu8
  // CHECK: @llvm.x86.avx512.mask.pavg.b.256
  return _mm256_maskz_avg_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_avg_epu16(__m128i __W, __mmask8 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.128
  return _mm_mask_avg_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_avg_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.128
  return _mm_maskz_avg_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_avg_epu16(__m256i __W, __mmask16 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.256
  return _mm256_mask_avg_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_avg_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_avg_epu16
  // CHECK: @llvm.x86.avx512.mask.pavg.w.256
  return _mm256_maskz_avg_epu16(__U,__A,__B); 
}
__m128i test_mm_maskz_max_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.128
  return _mm_maskz_max_epi8(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.128
  return _mm_mask_max_epi8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.256
  return _mm256_maskz_max_epi8(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi8
  // CHECK: @llvm.x86.avx512.mask.pmaxs.b.256
  return _mm256_mask_max_epi8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.128
  return _mm_maskz_max_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.128
  return _mm_mask_max_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.256
  return _mm256_maskz_max_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaxs.w.256
  return _mm256_mask_max_epi16(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epu8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.128
  return _mm_maskz_max_epu8(__M,__A,__B); 
}
__m128i test_mm_mask_max_epu8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.128
  return _mm_mask_max_epu8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.256
  return _mm256_maskz_max_epu8(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epu8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu8
  // CHECK: @llvm.x86.avx512.mask.pmaxu.b.256
  return _mm256_mask_max_epu8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epu16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.128
  return _mm_maskz_max_epu16(__M,__A,__B); 
}
__m128i test_mm_mask_max_epu16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.128
  return _mm_mask_max_epu16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.256
  return _mm256_maskz_max_epu16(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epu16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu16
  // CHECK: @llvm.x86.avx512.mask.pmaxu.w.256
  return _mm256_mask_max_epu16(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.128
  return _mm_maskz_min_epi8(__M,__A,__B); 
}
__m128i test_mm_mask_min_epi8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.128
  return _mm_mask_min_epi8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.256
  return _mm256_maskz_min_epi8(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epi8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi8
  // CHECK: @llvm.x86.avx512.mask.pmins.b.256
  return _mm256_mask_min_epi8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.128
  return _mm_maskz_min_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_min_epi16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.128
  return _mm_mask_min_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.256
  return _mm256_maskz_min_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epi16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi16
  // CHECK: @llvm.x86.avx512.mask.pmins.w.256
  return _mm256_mask_min_epi16(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.128
  return _mm_maskz_min_epu8(__M,__A,__B); 
}
__m128i test_mm_mask_min_epu8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.128
  return _mm_mask_min_epu8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.256
  return _mm256_maskz_min_epu8(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epu8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu8
  // CHECK: @llvm.x86.avx512.mask.pminu.b.256
  return _mm256_mask_min_epu8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.128
  return _mm_maskz_min_epu16(__M,__A,__B); 
}
__m128i test_mm_mask_min_epu16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.128
  return _mm_mask_min_epu16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.256
  return _mm256_maskz_min_epu16(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epu16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu16
  // CHECK: @llvm.x86.avx512.mask.pminu.w.256
  return _mm256_mask_min_epu16(__W,__M,__A,__B); 
}
__m128i test_mm_mask_shuffle_epi8(__m128i __W, __mmask16 __U, __m128i __A,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.128
  return _mm_mask_shuffle_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_shuffle_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.128
  return _mm_maskz_shuffle_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_shuffle_epi8(__m256i __W, __mmask32 __U, __m256i __A,        __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.256
  return _mm256_mask_shuffle_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_shuffle_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_epi8
  // CHECK: @llvm.x86.avx512.mask.pshuf.b.256
  return _mm256_maskz_shuffle_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epi8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.128
  return _mm_mask_subs_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.128
  return _mm_maskz_subs_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epi8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.256
  return _mm256_mask_subs_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epi8
  // CHECK: @llvm.x86.avx512.mask.psubs.b.256
  return _mm256_maskz_subs_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epi16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.128
  return _mm_mask_subs_epi16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.128
  return _mm_maskz_subs_epi16(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epi16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.256
  return _mm256_mask_subs_epi16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epi16
  // CHECK: @llvm.x86.avx512.mask.psubs.w.256
  return _mm256_maskz_subs_epi16(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epu8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.128
  return _mm_mask_subs_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.128
  return _mm_maskz_subs_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epu8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.256
  return _mm256_mask_subs_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epu8
  // CHECK: @llvm.x86.avx512.mask.psubus.b.256
  return _mm256_maskz_subs_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epu16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.128
  return _mm_mask_subs_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.128
  return _mm_maskz_subs_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epu16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.256
  return _mm256_mask_subs_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epu16
  // CHECK: @llvm.x86.avx512.mask.psubus.w.256
  return _mm256_maskz_subs_epu16(__U,__A,__B); 
}


__m128i test_mm_mask2_permutex2var_epi16(__m128i __A, __m128i __I, __mmask8 __U,            __m128i __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.hi.128
  return _mm_mask2_permutex2var_epi16(__A,__I,__U,__B); 
}
__m256i test_mm256_mask2_permutex2var_epi16(__m256i __A, __m256i __I,         __mmask16 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.hi.256
  return _mm256_mask2_permutex2var_epi16(__A,__I,__U,__B); 
}
__m128i test_mm_permutex2var_epi16(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.128
  return _mm_permutex2var_epi16(__A,__I,__B); 
}
__m128i test_mm_mask_permutex2var_epi16(__m128i __A, __mmask8 __U, __m128i __I,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.128
  return _mm_mask_permutex2var_epi16(__A,__U,__I,__B); 
}
__m128i test_mm_maskz_permutex2var_epi16(__mmask8 __U, __m128i __A, __m128i __I,            __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.hi.128
  return _mm_maskz_permutex2var_epi16(__U,__A,__I,__B); 
}

__m256i test_mm256_permutex2var_epi16(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.256
  return _mm256_permutex2var_epi16(__A,__I,__B); 
}
__m256i test_mm256_mask_permutex2var_epi16(__m256i __A, __mmask16 __U,        __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.hi.256
  return _mm256_mask_permutex2var_epi16(__A,__U,__I,__B); 
}
__m256i test_mm256_maskz_permutex2var_epi16(__mmask16 __U, __m256i __A,         __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.hi.256
  return _mm256_maskz_permutex2var_epi16(__U,__A,__I,__B); 
}
__m128i test_mm_mask_maddubs_epi16(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_maddubs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddubs.w.128
  return _mm_mask_maddubs_epi16(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_maddubs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_maddubs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddubs.w.128
  return _mm_maskz_maddubs_epi16(__U, __X, __Y); 
}

__m256i test_mm256_mask_maddubs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_maddubs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddubs.w.256
  return _mm256_mask_maddubs_epi16(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_maddubs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_maddubs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddubs.w.256
  return _mm256_maskz_maddubs_epi16(__U, __X, __Y); 
}

__m128i test_mm_mask_madd_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_madd_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddw.d.128
  return _mm_mask_madd_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_madd_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_madd_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddw.d.128
  return _mm_maskz_madd_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_madd_epi16(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_madd_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddw.d.256
  return _mm256_mask_madd_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_madd_epi16(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_madd_epi16
  // CHECK: @llvm.x86.avx512.mask.pmaddw.d.256
  return _mm256_maskz_madd_epi16(__U, __A, __B); 
}

__m128i test_mm_cvtsepi16_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.128
  return _mm_cvtsepi16_epi8(__A); 
}

__m128i test_mm_mask_cvtsepi16_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.128
  return _mm_mask_cvtsepi16_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi16_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.128
  return _mm_maskz_cvtsepi16_epi8(__M, __A); 
}

__m128i test_mm256_cvtsepi16_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.256
  return _mm256_cvtsepi16_epi8(__A); 
}

__m128i test_mm256_mask_cvtsepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.256
  return _mm256_mask_cvtsepi16_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.256
  return _mm256_maskz_cvtsepi16_epi8(__M, __A); 
}

__m128i test_mm_cvtusepi16_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.128
  return _mm_cvtusepi16_epi8(__A); 
}

__m128i test_mm_mask_cvtusepi16_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.128
  return _mm_mask_cvtusepi16_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi16_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.128
  return _mm_maskz_cvtusepi16_epi8(__M, __A); 
}

__m128i test_mm256_cvtusepi16_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.256
  return _mm256_cvtusepi16_epi8(__A); 
}

__m128i test_mm256_mask_cvtusepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.256
  return _mm256_mask_cvtusepi16_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.256
  return _mm256_maskz_cvtusepi16_epi8(__M, __A); 
}

__m128i test_mm_cvtepi16_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.128
  return _mm_cvtepi16_epi8(__A); 
}

__m128i test_mm_mask_cvtepi16_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.128
  return _mm_mask_cvtepi16_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi16_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.128
  return _mm_maskz_cvtepi16_epi8(__M, __A); 
}

__m128i test_mm256_cvtepi16_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.256
  return _mm256_cvtepi16_epi8(__A); 
}

__m128i test_mm256_mask_cvtepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.256
  return _mm256_mask_cvtepi16_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.256
  return _mm256_maskz_cvtepi16_epi8(__M, __A); 
}

__m128i test_mm_mask_mulhrs_epi16(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmul.hr.sw.128
  return _mm_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_mulhrs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmul.hr.sw.128
  return _mm_maskz_mulhrs_epi16(__U, __X, __Y); 
}

__m256i test_mm256_mask_mulhrs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmul.hr.sw.256
  return _mm256_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_mulhrs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.avx512.mask.pmul.hr.sw.256
  return _mm256_maskz_mulhrs_epi16(__U, __X, __Y); 
}

__m128i test_mm_mask_mulhi_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mulhi_epu16
  // CHECK: @llvm.x86.avx512.mask.pmulhu.w.128
  return _mm_mask_mulhi_epu16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_mulhi_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mulhi_epu16
  // CHECK: @llvm.x86.avx512.mask.pmulhu.w.128
  return _mm_maskz_mulhi_epu16(__U, __A, __B); 
}

__m256i test_mm256_mask_mulhi_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mulhi_epu16
  // CHECK: @llvm.x86.avx512.mask.pmulhu.w.256
  return _mm256_mask_mulhi_epu16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_mulhi_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mulhi_epu16
  // CHECK: @llvm.x86.avx512.mask.pmulhu.w.256
  return _mm256_maskz_mulhi_epu16(__U, __A, __B); 
}

__m128i test_mm_mask_mulhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mulhi_epi16
  // CHECK: @llvm.x86.avx512.mask.pmulh.w.128
  return _mm_mask_mulhi_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_mulhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mulhi_epi16
  // CHECK: @llvm.x86.avx512.mask.pmulh.w.128
  return _mm_maskz_mulhi_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_mulhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mulhi_epi16
  // CHECK: @llvm.x86.avx512.mask.pmulh.w.256
  return _mm256_mask_mulhi_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_mulhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mulhi_epi16
  // CHECK: @llvm.x86.avx512.mask.pmulh.w.256
  return _mm256_maskz_mulhi_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi8
  // CHECK: @llvm.x86.avx512.mask.punpckhb.w.128
  return _mm_mask_unpackhi_epi8(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi8
  // CHECK: @llvm.x86.avx512.mask.punpckhb.w.128
  return _mm_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi8
  // CHECK: @llvm.x86.avx512.mask.punpckhb.w.256
  return _mm256_mask_unpackhi_epi8(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi8
  // CHECK: @llvm.x86.avx512.mask.punpckhb.w.256
  return _mm256_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi16
  // CHECK: @llvm.x86.avx512.mask.punpckhw.d.128
  return _mm_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi16
  // CHECK: @llvm.x86.avx512.mask.punpckhw.d.128
  return _mm_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi16
  // CHECK: @llvm.x86.avx512.mask.punpckhw.d.256
  return _mm256_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi16
  // CHECK: @llvm.x86.avx512.mask.punpckhw.d.256
  return _mm256_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi8
  // CHECK: @llvm.x86.avx512.mask.punpcklb.w.128
  return _mm_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi8
  // CHECK: @llvm.x86.avx512.mask.punpcklb.w.128
  return _mm_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi8
  // CHECK: @llvm.x86.avx512.mask.punpcklb.w.256
  return _mm256_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi8
  // CHECK: @llvm.x86.avx512.mask.punpcklb.w.256
  return _mm256_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi16
  // CHECK: @llvm.x86.avx512.mask.punpcklw.d.128
  return _mm_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi16
  // CHECK: @llvm.x86.avx512.mask.punpcklw.d.128
  return _mm_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi16
  // CHECK: @llvm.x86.avx512.mask.punpcklw.d.256
  return _mm256_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi16
  // CHECK: @llvm.x86.avx512.mask.punpcklw.d.256
  return _mm256_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_cvtepi8_epi16(__m128i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.w.128
  return _mm_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.w.128
  return _mm_maskz_cvtepi8_epi16(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi16(__m256i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.w.256
  return _mm256_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.w.256
  return _mm256_maskz_cvtepi8_epi16(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.128
  return _mm_mask_cvtepu8_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.128
  return _mm_maskz_cvtepu8_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepu8_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.256
  return _mm256_mask_cvtepu8_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.256
  return _mm256_maskz_cvtepu8_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.128
  return _mm_mask_cvtepu8_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.128
  return _mm_maskz_cvtepu8_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepu8_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.256
  return _mm256_mask_cvtepu8_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.256
  return _mm256_maskz_cvtepu8_epi64(__U, __A); 
}

__m128i test_mm_mask_cvtepu32_epi64(__m128i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_mask_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.128
  return _mm_mask_cvtepu32_epi64(__W, __U, __X); 
}

__m128i test_mm_maskz_cvtepu32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.128
  return _mm_maskz_cvtepu32_epi64(__U, __X); 
}

__m256i test_mm256_mask_cvtepu32_epi64(__m256i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.256
  return _mm256_mask_cvtepu32_epi64(__W, __U, __X); 
}

__m256i test_mm256_maskz_cvtepu32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.256
  return _mm256_maskz_cvtepu32_epi64(__U, __X); 
}

__m128i test_mm_mask_cvtepu16_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.128
  return _mm_mask_cvtepu16_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.128
  return _mm_maskz_cvtepu16_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepu16_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.256
  return _mm256_mask_cvtepu16_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.256
  return _mm256_maskz_cvtepu16_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepu16_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.128
  return _mm_mask_cvtepu16_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.128
  return _mm_maskz_cvtepu16_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepu16_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.256
  return _mm256_mask_cvtepu16_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.256
  return _mm256_maskz_cvtepu16_epi64(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi16(__m128i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.w.128
  return _mm_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.w.128
  return _mm_maskz_cvtepu8_epi16(__U, __A); 
}

__m256i test_mm256_mask_cvtepu8_epi16(__m256i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.w.256
  return _mm256_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu8_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.w.256
  return _mm256_maskz_cvtepu8_epi16(__U, __A); 
}