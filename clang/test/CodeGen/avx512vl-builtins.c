// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__mmask8 test_mm256_cmpeq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.256
  return (__mmask8)_mm256_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.256
  return (__mmask8)_mm256_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.128
  return (__mmask8)_mm_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.128
  return (__mmask8)_mm_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.256
  return (__mmask8)_mm256_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.256
  return (__mmask8)_mm256_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.128
  return (__mmask8)_mm_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.128
  return (__mmask8)_mm_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.256
  return (__mmask8)_mm256_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.256
  return (__mmask8)_mm256_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.128
  return (__mmask8)_mm_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.128
  return (__mmask8)_mm_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.256
  return (__mmask8)_mm256_cmpgt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.256
  return (__mmask8)_mm256_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.128
  return (__mmask8)_mm_cmpgt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.128
  return (__mmask8)_mm_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 0, i8 -1)
  return (__mmask8)_mm_cmpeq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 0, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 0, i8 -1)
  return (__mmask8)_mm_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 0, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm256_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm256_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm256_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm256_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm256_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm256_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmp_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epi32_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epi32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epi64_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epi64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epi32_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epi32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epi64_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epi64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epu32_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.128(<4 x i32> {{.*}}, <4 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epu32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm_cmp_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm_cmp_epu64_mask(__a, __b, 7);
}

__mmask8 test_mm_mask_cmp_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.128(<2 x i64> {{.*}}, <2 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm_mask_cmp_epu64_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epu32_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.256(<8 x i32> {{.*}}, <8 x i32> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epu32_mask(__u, __a, __b, 7);
}

__mmask8 test_mm256_cmp_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 -1)
  return (__mmask8)_mm256_cmp_epu64_mask(__a, __b, 7);
}

__mmask8 test_mm256_mask_cmp_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.256(<4 x i64> {{.*}}, <4 x i64> {{.*}}, i32 7, i8 {{.*}})
  return (__mmask8)_mm256_mask_cmp_epu64_mask(__u, __a, __b, 7);
}

__m512i test_mm512_maskz_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_maskz_andnot_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_mask_andnot_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_andnot_epi32(__A,__B);
}

__m512i test_mm512_maskz_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_maskz_andnot_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_mask_andnot_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_andnot_epi64(__A,__B);
}

__m256i test_mm256_mask_add_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.256
  return _mm256_mask_add_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_add_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.256
  return _mm256_maskz_add_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_add_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.256
  return _mm256_mask_add_epi64(__W,__U,__A,__B);
}

__m256i test_mm256_maskz_add_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.256
  return _mm256_maskz_add_epi64 (__U,__A,__B);
}

__m256i test_mm256_mask_sub_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.256
  return _mm256_mask_sub_epi32 (__W,__U,__A,__B);
}

__m256i test_mm256_maskz_sub_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.256
  return _mm256_maskz_sub_epi32 (__U,__A,__B);
}

__m256i test_mm256_mask_sub_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.256
  return _mm256_mask_sub_epi64 (__W,__U,__A,__B);
}

__m256i test_mm256_maskz_sub_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.256
  return _mm256_maskz_sub_epi64 (__U,__A,__B);
}

__m128i test_mm_mask_add_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.128
  return _mm_mask_add_epi32(__W,__U,__A,__B);
}


__m128i test_mm_maskz_add_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.128
  return _mm_maskz_add_epi32 (__U,__A,__B);
}

__m128i test_mm_mask_add_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
//CHECK-LABEL: @test_mm_mask_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.128
  return _mm_mask_add_epi64 (__W,__U,__A,__B);
}

__m128i test_mm_maskz_add_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.128
  return _mm_maskz_add_epi64 (__U,__A,__B);
}

__m128i test_mm_mask_sub_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.128
  return _mm_mask_sub_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_sub_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.128
  return _mm_maskz_sub_epi32(__U, __A, __B);
}

__m128i test_mm_mask_sub_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.128
  return _mm_mask_sub_epi64 (__W, __U, __A, __B);
}

__m128i test_mm_maskz_sub_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.128
  return _mm_maskz_sub_epi64 (__U, __A, __B);
}

__m256i test_mm256_mask_mul_epi32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y) {
  //CHECK-LABEL: @test_mm256_mask_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.256
  return _mm256_mask_mul_epi32(__W, __M, __X, __Y);
}

__m256i test_mm256_maskz_mul_epi32 (__mmask8 __M, __m256i __X, __m256i __Y) {
  //CHECK-LABEL: @test_mm256_maskz_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.256
  return _mm256_maskz_mul_epi32(__M, __X, __Y);
}


__m128i test_mm_mask_mul_epi32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y) {
  //CHECK-LABEL: @test_mm_mask_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.128
  return _mm_mask_mul_epi32(__W, __M, __X, __Y);
}

__m128i test_mm_maskz_mul_epi32 (__mmask8 __M, __m128i __X, __m128i __Y) {
  //CHECK-LABEL: @test_mm_maskz_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.128
  return _mm_maskz_mul_epi32(__M, __X, __Y);
}

__m256i test_mm256_mask_mul_epu32 (__m256i __W, __mmask8 __M, __m256i __X,
           __m256i __Y) {
  //CHECK-LABEL: @test_mm256_mask_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.256
  return _mm256_mask_mul_epu32(__W, __M, __X, __Y);
}

__m256i test_mm256_maskz_mul_epu32 (__mmask8 __M, __m256i __X, __m256i __Y) {
  //CHECK-LABEL: @test_mm256_maskz_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.256
  return _mm256_maskz_mul_epu32(__M, __X, __Y);
}

__m128i test_mm_mask_mul_epu32 (__m128i __W, __mmask8 __M, __m128i __X,
        __m128i __Y) {
  //CHECK-LABEL: @test_mm_mask_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.128
  return _mm_mask_mul_epu32(__W, __M, __X, __Y);
}

__m128i test_mm_maskz_mul_epu32 (__mmask8 __M, __m128i __X, __m128i __Y) {
  //CHECK-LABEL: @test_mm_maskz_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.128
  return _mm_maskz_mul_epu32(__M, __X, __Y);
}

__m128i test_mm_maskz_mullo_epi32 (__mmask8 __M, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.128
  return _mm_maskz_mullo_epi32(__M, __A, __B);
}

__m128i test_mm_mask_mullo_epi32 (__m128i __W, __mmask8 __M, __m128i __A,
          __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.128
  return _mm_mask_mullo_epi32(__W, __M, __A, __B);
}

__m256i test_mm256_maskz_mullo_epi32 (__mmask8 __M, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.256
  return _mm256_maskz_mullo_epi32(__M, __A, __B);
}

__m256i test_mm256_mask_mullo_epi32 (__m256i __W, __mmask8 __M, __m256i __A,
       __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.256
  return _mm256_mask_mullo_epi32(__W, __M, __A, __B);
}

__m256i test_mm256_mask_and_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.256
  return _mm256_mask_and_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.256
  return _mm256_maskz_and_epi32(__U, __A, __B);
}

__m128i test_mm_mask_and_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.128
  return _mm_mask_and_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_and_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi32
  //CHECK: @llvm.x86.avx512.mask.pand.d.128
  return _mm_maskz_and_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_andnot_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.256
  return _mm256_mask_andnot_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.256
  return _mm256_maskz_andnot_epi32(__U, __A, __B);
}

__m128i test_mm_mask_andnot_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.128
  return _mm_mask_andnot_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.128
  return _mm_maskz_andnot_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_or_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.256
  return _mm256_mask_or_epi32(__W, __U, __A, __B);
}

 __m256i test_mm256_maskz_or_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.256
  return _mm256_maskz_or_epi32(__U, __A, __B);
}

 __m128i test_mm_mask_or_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.128
  return _mm_mask_or_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_or_epi32
  //CHECK: @llvm.x86.avx512.mask.por.d.128
  return _mm_maskz_or_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_xor_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.256
  return _mm256_mask_xor_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.256
  return _mm256_maskz_xor_epi32(__U, __A, __B);
}

__m128i test_mm_mask_xor_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.128
  return _mm_mask_xor_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi32
  //CHECK: @llvm.x86.avx512.mask.pxor.d.128
  return _mm_maskz_xor_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_and_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.256
  return _mm256_mask_and_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.256
  return _mm256_maskz_and_epi64(__U, __A, __B);
}

__m128i test_mm_mask_and_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.128
  return _mm_mask_and_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_and_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi64
  //CHECK: @llvm.x86.avx512.mask.pand.q.128
  return _mm_maskz_and_epi64(__U, __A, __B);
}

__m256i test_mm256_mask_andnot_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.256
  return _mm256_mask_andnot_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.256
  return _mm256_maskz_andnot_epi64(__U, __A, __B);
}

__m128i test_mm_mask_andnot_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.128
  return _mm_mask_andnot_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.128
  return _mm_maskz_andnot_epi64(__U, __A, __B);
}

__m256i test_mm256_mask_or_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.256
  return _mm256_mask_or_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_or_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.256
  return _mm256_maskz_or_epi64(__U, __A, __B);
}

__m128i test_mm_mask_or_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.128
  return _mm_mask_or_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
//CHECK-LABEL: @test_mm_maskz_or_epi64
  //CHECK: @llvm.x86.avx512.mask.por.q.128
  return _mm_maskz_or_epi64( __U, __A, __B);
}

__m256i test_mm256_mask_xor_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.256
  return _mm256_mask_xor_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.256
  return _mm256_maskz_xor_epi64(__U, __A, __B);
}

__m128i test_mm_mask_xor_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.128
  return _mm_mask_xor_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi64
  //CHECK: @llvm.x86.avx512.mask.pxor.q.128
  return _mm_maskz_xor_epi64( __U, __A, __B);
}

__mmask8 test_mm256_cmp_ps_mask(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.256
  return (__mmask8)_mm256_cmp_ps_mask(__A, __B, 0);
}

__mmask8 test_mm256_mask_cmp_ps_mask(__mmask8 m, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.256
  return _mm256_mask_cmp_ps_mask(m, __A, __B, 0);
}

__mmask8 test_mm128_cmp_ps_mask(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm128_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.128
  return (__mmask8)_mm128_cmp_ps_mask(__A, __B, 0);
}

__mmask8 test_mm128_mask_cmp_ps_mask(__mmask8 m, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm128_mask_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.128
  return _mm128_mask_cmp_ps_mask(m, __A, __B, 0);
}

__mmask8 test_mm256_cmp_pd_mask(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.256
  return (__mmask8)_mm256_cmp_pd_mask(__A, __B, 0);
}

__mmask8 test_mm256_mask_cmp_pd_mask(__mmask8 m, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.256
  return _mm256_mask_cmp_pd_mask(m, __A, __B, 0);
}

__mmask8 test_mm128_cmp_pd_mask(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm128_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.128
  return (__mmask8)_mm128_cmp_pd_mask(__A, __B, 0);
}

__mmask8 test_mm128_mask_cmp_pd_mask(__mmask8 m, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm128_mask_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.128
  return _mm128_mask_cmp_pd_mask(m, __A, __B, 0);
}


//igorb

__m128d test_mm_mask_fmadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.128
  return _mm_mask_fmadd_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask_fmsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.128
  return _mm_mask_fmsub_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask3_fmadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.128
  return _mm_mask3_fmadd_pd(__A, __B, __C, __U);
}

__m128d test_mm_mask3_fnmadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.128
  return _mm_mask3_fnmadd_pd(__A, __B, __C, __U);
}

__m128d test_mm_maskz_fmadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.128
  return _mm_maskz_fmadd_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fmsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.128
  return _mm_maskz_fmsub_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fnmadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.128
  return _mm_maskz_fnmadd_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fnmsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.128
  return _mm_maskz_fnmsub_pd(__U, __A, __B, __C);
}

__m256d test_mm256_mask_fmadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.256
  return _mm256_mask_fmadd_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask_fmsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.256
  return _mm256_mask_fmsub_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask3_fmadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.256
  return _mm256_mask3_fmadd_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask3_fnmadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.256
  return _mm256_mask3_fnmadd_pd(__A, __B, __C, __U);
}

__m256d test_mm256_maskz_fmadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.256
  return _mm256_maskz_fmadd_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fmsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.256
  return _mm256_maskz_fmsub_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fnmadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.256
  return _mm256_maskz_fnmadd_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fnmsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.256
  return _mm256_maskz_fnmsub_pd(__U, __A, __B, __C);
}

__m128 test_mm_mask_fmadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.128
  return _mm_mask_fmadd_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask_fmsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.128
  return _mm_mask_fmsub_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask3_fmadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.128
  return _mm_mask3_fmadd_ps(__A, __B, __C, __U);
}

__m128 test_mm_mask3_fnmadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.128
  return _mm_mask3_fnmadd_ps(__A, __B, __C, __U);
}

__m128 test_mm_maskz_fmadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.128
  return _mm_maskz_fmadd_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fmsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.128
  return _mm_maskz_fmsub_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fnmadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.128
  return _mm_maskz_fnmadd_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fnmsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fnmsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.128
  return _mm_maskz_fnmsub_ps(__U, __A, __B, __C);
}

__m256 test_mm256_mask_fmadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.256
  return _mm256_mask_fmadd_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask_fmsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.256
  return _mm256_mask_fmsub_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask3_fmadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.256
  return _mm256_mask3_fmadd_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask3_fnmadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.256
  return _mm256_mask3_fnmadd_ps(__A, __B, __C, __U);
}

__m256 test_mm256_maskz_fmadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.256
  return _mm256_maskz_fmadd_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fmsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.256
  return _mm256_maskz_fmsub_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fnmadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.256
  return _mm256_maskz_fnmadd_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fnmsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fnmsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.256
  return _mm256_maskz_fnmsub_ps(__U, __A, __B, __C);
}

__m128d test_mm_mask_fmaddsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.128
  return _mm_mask_fmaddsub_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask_fmsubadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.128
  return _mm_mask_fmsubadd_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask3_fmaddsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.pd.128
  return _mm_mask3_fmaddsub_pd(__A, __B, __C, __U);
}

__m128d test_mm_maskz_fmaddsub_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.128
  return _mm_maskz_fmaddsub_pd(__U, __A, __B, __C);
}

__m128d test_mm_maskz_fmsubadd_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.128
  return _mm_maskz_fmsubadd_pd(__U, __A, __B, __C);
}

__m256d test_mm256_mask_fmaddsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.256
  return _mm256_mask_fmaddsub_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask_fmsubadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.256
  return _mm256_mask_fmsubadd_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask3_fmaddsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.pd.256
  return _mm256_mask3_fmaddsub_pd(__A, __B, __C, __U);
}

__m256d test_mm256_maskz_fmaddsub_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.256
  return _mm256_maskz_fmaddsub_pd(__U, __A, __B, __C);
}

__m256d test_mm256_maskz_fmsubadd_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.256
  return _mm256_maskz_fmsubadd_pd(__U, __A, __B, __C);
}

__m128 test_mm_mask_fmaddsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.128
  return _mm_mask_fmaddsub_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask_fmsubadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.128
  return _mm_mask_fmsubadd_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask3_fmaddsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.ps.128
  return _mm_mask3_fmaddsub_ps(__A, __B, __C, __U);
}

__m128 test_mm_maskz_fmaddsub_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.128
  return _mm_maskz_fmaddsub_ps(__U, __A, __B, __C);
}

__m128 test_mm_maskz_fmsubadd_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_maskz_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.128
  return _mm_maskz_fmsubadd_ps(__U, __A, __B, __C);
}

__m256 test_mm256_mask_fmaddsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.256
  return _mm256_mask_fmaddsub_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask_fmsubadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.256
  return _mm256_mask_fmsubadd_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask3_fmaddsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.ps.256
  return _mm256_mask3_fmaddsub_ps(__A, __B, __C, __U);
}

__m256 test_mm256_maskz_fmaddsub_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.256
  return _mm256_maskz_fmaddsub_ps(__U, __A, __B, __C);
}

__m256 test_mm256_maskz_fmsubadd_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_maskz_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.256
  return _mm256_maskz_fmsubadd_ps(__U, __A, __B, __C);
}

__m128d test_mm_mask3_fmsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.pd.128
  return _mm_mask3_fmsub_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask3_fmsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.pd.256
  return _mm256_mask3_fmsub_pd(__A, __B, __C, __U);
}

__m128 test_mm_mask3_fmsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.ps.128
  return _mm_mask3_fmsub_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask3_fmsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.ps.256
  return _mm256_mask3_fmsub_ps(__A, __B, __C, __U);
}

__m128d test_mm_mask3_fmsubadd_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.pd.128
  return _mm_mask3_fmsubadd_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask3_fmsubadd_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.pd.256
  return _mm256_mask3_fmsubadd_pd(__A, __B, __C, __U);
}

__m128 test_mm_mask3_fmsubadd_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.ps.128
  return _mm_mask3_fmsubadd_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask3_fmsubadd_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.ps.256
  return _mm256_mask3_fmsubadd_ps(__A, __B, __C, __U);
}

__m128d test_mm_mask_fnmadd_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.pd.128
  return _mm_mask_fnmadd_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask_fnmadd_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.pd.256
  return _mm256_mask_fnmadd_pd(__A, __U, __B, __C);
}

__m128 test_mm_mask_fnmadd_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.ps.128
  return _mm_mask_fnmadd_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask_fnmadd_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.ps.256
  return _mm256_mask_fnmadd_ps(__A, __U, __B, __C);
}

__m128d test_mm_mask_fnmsub_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128d __C) {
  // CHECK-LABEL: @test_mm_mask_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.pd.128
  return _mm_mask_fnmsub_pd(__A, __U, __B, __C);
}

__m128d test_mm_mask3_fnmsub_pd(__m128d __A, __m128d __B, __m128d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.pd.128
  return _mm_mask3_fnmsub_pd(__A, __B, __C, __U);
}

__m256d test_mm256_mask_fnmsub_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256d __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.pd.256
  return _mm256_mask_fnmsub_pd(__A, __U, __B, __C);
}

__m256d test_mm256_mask3_fnmsub_pd(__m256d __A, __m256d __B, __m256d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.pd.256
  return _mm256_mask3_fnmsub_pd(__A, __B, __C, __U);
}

__m128 test_mm_mask_fnmsub_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128 __C) {
  // CHECK-LABEL: @test_mm_mask_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.ps.128
  return _mm_mask_fnmsub_ps(__A, __U, __B, __C);
}

__m128 test_mm_mask3_fnmsub_ps(__m128 __A, __m128 __B, __m128 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm_mask3_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.ps.128
  return _mm_mask3_fnmsub_ps(__A, __B, __C, __U);
}

__m256 test_mm256_mask_fnmsub_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256 __C) {
  // CHECK-LABEL: @test_mm256_mask_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.ps.256
  return _mm256_mask_fnmsub_ps(__A, __U, __B, __C);
}

__m256 test_mm256_mask3_fnmsub_ps(__m256 __A, __m256 __B, __m256 __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm256_mask3_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.ps.256 
  return _mm256_mask3_fnmsub_ps(__A, __B, __C, __U);
}

__m128d test_mm_mask_add_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_add_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.128
  return _mm_mask_add_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_add_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_add_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.128
  return _mm_maskz_add_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_add_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_add_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.256
  return _mm256_mask_add_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_add_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_add_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.256
  return _mm256_maskz_add_pd(__U,__A,__B); 
}
__m128 test_mm_mask_add_ps(__m128 __W, __mmask16 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_add_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.128
  return _mm_mask_add_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_add_ps(__mmask16 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_add_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.128
  return _mm_maskz_add_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_add_ps(__m256 __W, __mmask16 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_add_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.256
  return _mm256_mask_add_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_add_ps(__mmask16 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_add_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.256
  return _mm256_maskz_add_ps(__U,__A,__B); 
}
__m128i test_mm_mask_blend_epi32(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi32
  // CHECK: @llvm.x86.avx512.mask.blend.d.128
  return _mm_mask_blend_epi32(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi32(__mmask8 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi32
  // CHECK: @llvm.x86.avx512.mask.blend.d.256
  return _mm256_mask_blend_epi32(__U,__A,__W); 
}
__m128d test_mm_mask_blend_pd(__mmask8 __U, __m128d __A, __m128d __W) {
  // CHECK-LABEL: @test_mm_mask_blend_pd
  // CHECK: @llvm.x86.avx512.mask.blend.pd.128
  return _mm_mask_blend_pd(__U,__A,__W); 
}
__m256d test_mm256_mask_blend_pd(__mmask8 __U, __m256d __A, __m256d __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_pd
  // CHECK: @llvm.x86.avx512.mask.blend.pd.256
  return _mm256_mask_blend_pd(__U,__A,__W); 
}
__m128 test_mm_mask_blend_ps(__mmask8 __U, __m128 __A, __m128 __W) {
  // CHECK-LABEL: @test_mm_mask_blend_ps
  // CHECK: @llvm.x86.avx512.mask.blend.ps.128
  return _mm_mask_blend_ps(__U,__A,__W); 
}
__m256 test_mm256_mask_blend_ps(__mmask8 __U, __m256 __A, __m256 __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_ps
  // CHECK: @llvm.x86.avx512.mask.blend.ps.256
  return _mm256_mask_blend_ps(__U,__A,__W); 
}
__m128i test_mm_mask_blend_epi64(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi64
  // CHECK: @llvm.x86.avx512.mask.blend.q.128
  return _mm_mask_blend_epi64(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi64(__mmask8 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi64
  // CHECK: @llvm.x86.avx512.mask.blend.q.256
  return _mm256_mask_blend_epi64(__U,__A,__W); 
}
__m128d test_mm_mask_compress_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress.pd.128
  return _mm_mask_compress_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_compress_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress.pd.128
  return _mm_maskz_compress_pd(__U,__A); 
}
__m256d test_mm256_mask_compress_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress.pd.256
  return _mm256_mask_compress_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_compress_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_pd
  // CHECK: @llvm.x86.avx512.mask.compress.pd.256
  return _mm256_maskz_compress_pd(__U,__A); 
}
__m128i test_mm_mask_compress_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress.q.128
  return _mm_mask_compress_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_compress_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress.q.128
  return _mm_maskz_compress_epi64(__U,__A); 
}
__m256i test_mm256_mask_compress_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress.q.256
  return _mm256_mask_compress_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_compress_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_epi64
  // CHECK: @llvm.x86.avx512.mask.compress.q.256
  return _mm256_maskz_compress_epi64(__U,__A); 
}
__m128 test_mm_mask_compress_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress.ps.128
  return _mm_mask_compress_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_compress_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress.ps.128
  return _mm_maskz_compress_ps(__U,__A); 
}
__m256 test_mm256_mask_compress_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress.ps.256
  return _mm256_mask_compress_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_compress_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_ps
  // CHECK: @llvm.x86.avx512.mask.compress.ps.256
  return _mm256_maskz_compress_ps(__U,__A); 
}
__m128i test_mm_mask_compress_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress.d.128
  return _mm_mask_compress_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_compress_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress.d.128
  return _mm_maskz_compress_epi32(__U,__A); 
}
__m256i test_mm256_mask_compress_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress.d.256
  return _mm256_mask_compress_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_compress_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_compress_epi32
  // CHECK: @llvm.x86.avx512.mask.compress.d.256
  return _mm256_maskz_compress_epi32(__U,__A); 
}
void test_mm_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_pd
  // CHECK: @llvm.x86.avx512.mask.compress.store.pd.128
  return _mm_mask_compressstoreu_pd(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_pd
  // CHECK: @llvm.x86.avx512.mask.compress.store.pd.256
  return _mm256_mask_compressstoreu_pd(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_epi64
  // CHECK: @llvm.x86.avx512.mask.compress.store.q.128
  return _mm_mask_compressstoreu_epi64(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_epi64
  // CHECK: @llvm.x86.avx512.mask.compress.store.q.256
  return _mm256_mask_compressstoreu_epi64(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_ps
  // CHECK: @llvm.x86.avx512.mask.compress.store.ps.128
  return _mm_mask_compressstoreu_ps(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_ps
  // CHECK: @llvm.x86.avx512.mask.compress.store.ps.256
  return _mm256_mask_compressstoreu_ps(__P,__U,__A); 
}
void test_mm_mask_compressstoreu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_compressstoreu_epi32
  // CHECK: @llvm.x86.avx512.mask.compress.store.d.128
  return _mm_mask_compressstoreu_epi32(__P,__U,__A); 
}
void test_mm256_mask_compressstoreu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_compressstoreu_epi32
  // CHECK: @llvm.x86.avx512.mask.compress.store.d.256
  return _mm256_mask_compressstoreu_epi32(__P,__U,__A); 
}
__m128d test_mm_mask_cvtepi32_pd(__m128d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtdq2pd.128
  return _mm_mask_cvtepi32_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_cvtepi32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtdq2pd.128
  return _mm_maskz_cvtepi32_pd(__U,__A); 
}
__m256d test_mm256_mask_cvtepi32_pd(__m256d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtdq2pd.256
  return _mm256_mask_cvtepi32_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_cvtepi32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtdq2pd.256
  return _mm256_maskz_cvtepi32_pd(__U,__A); 
}
__m128 test_mm_mask_cvtepi32_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtdq2ps.128
  return _mm_mask_cvtepi32_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_cvtepi32_ps(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtdq2ps.128
  return _mm_maskz_cvtepi32_ps(__U,__A); 
}
__m256 test_mm256_mask_cvtepi32_ps(__m256 __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtdq2ps.256
  return _mm256_mask_cvtepi32_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_cvtepi32_ps(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtdq2ps.256
  return _mm256_maskz_cvtepi32_ps(__U,__A); 
}
__m128i test_mm_mask_cvtpd_epi32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.128
  return _mm_mask_cvtpd_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtpd_epi32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.128
  return _mm_maskz_cvtpd_epi32(__U,__A); 
}
__m128i test_mm256_mask_cvtpd_epi32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.256
  return _mm256_mask_cvtpd_epi32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvtpd_epi32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2dq.256
  return _mm256_maskz_cvtpd_epi32(__U,__A); 
}
__m128 test_mm_mask_cvtpd_ps(__m128 __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps
  return _mm_mask_cvtpd_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_cvtpd_ps(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps
  return _mm_maskz_cvtpd_ps(__U,__A); 
}
__m128 test_mm256_mask_cvtpd_ps(__m128 __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.256
  return _mm256_mask_cvtpd_ps(__W,__U,__A); 
}
__m128 test_mm256_maskz_cvtpd_ps(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_ps
  // CHECK: @llvm.x86.avx512.mask.cvtpd2ps.256
  return _mm256_maskz_cvtpd_ps(__U,__A); 
}
__m128i test_mm_cvtpd_epu32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.128
  return _mm_cvtpd_epu32(__A); 
}
__m128i test_mm_mask_cvtpd_epu32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.128
  return _mm_mask_cvtpd_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtpd_epu32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.128
  return _mm_maskz_cvtpd_epu32(__U,__A); 
}
__m128i test_mm256_cvtpd_epu32(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.256
  return _mm256_cvtpd_epu32(__A); 
}
__m128i test_mm256_mask_cvtpd_epu32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.256
  return _mm256_mask_cvtpd_epu32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvtpd_epu32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtpd2udq.256
  return _mm256_maskz_cvtpd_epu32(__U,__A); 
}
__m128i test_mm_mask_cvtps_epi32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.128
  return _mm_mask_cvtps_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtps_epi32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.128
  return _mm_maskz_cvtps_epi32(__U,__A); 
}
__m256i test_mm256_mask_cvtps_epi32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.256
  return _mm256_mask_cvtps_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvtps_epi32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvtps2dq.256
  return _mm256_maskz_cvtps_epi32(__U,__A); 
}
__m128d test_mm_mask_cvtps_pd(__m128d __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.128
  return _mm_mask_cvtps_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_cvtps_pd(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.128
  return _mm_maskz_cvtps_pd(__U,__A); 
}
__m256d test_mm256_mask_cvtps_pd(__m256d __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.256
  return _mm256_mask_cvtps_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_cvtps_pd(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_pd
  // CHECK: @llvm.x86.avx512.mask.cvtps2pd.256
  return _mm256_maskz_cvtps_pd(__U,__A); 
}
__m128i test_mm_cvtps_epu32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.128
  return _mm_cvtps_epu32(__A); 
}
__m128i test_mm_mask_cvtps_epu32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.128
  return _mm_mask_cvtps_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvtps_epu32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.128
  return _mm_maskz_cvtps_epu32(__U,__A); 
}
__m256i test_mm256_cvtps_epu32(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.256
  return _mm256_cvtps_epu32(__A); 
}
__m256i test_mm256_mask_cvtps_epu32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.256
  return _mm256_mask_cvtps_epu32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvtps_epu32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvtps2udq.256
  return _mm256_maskz_cvtps_epu32(__U,__A); 
}
__m128i test_mm_mask_cvttpd_epi32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvttpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.128
  return _mm_mask_cvttpd_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttpd_epi32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.128
  return _mm_maskz_cvttpd_epi32(__U,__A); 
}
__m128i test_mm256_mask_cvttpd_epi32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.256
  return _mm256_mask_cvttpd_epi32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvttpd_epi32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttpd_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2dq.256
  return _mm256_maskz_cvttpd_epi32(__U,__A); 
}
__m128i test_mm_cvttpd_epu32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.128
  return _mm_cvttpd_epu32(__A); 
}
__m128i test_mm_mask_cvttpd_epu32(__m128i __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.128
  return _mm_mask_cvttpd_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttpd_epu32(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.128
  return _mm_maskz_cvttpd_epu32(__U,__A); 
}
__m128i test_mm256_cvttpd_epu32(__m256d __A) {
  // CHECK-LABEL: @test_mm256_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.256
  return _mm256_cvttpd_epu32(__A); 
}
__m128i test_mm256_mask_cvttpd_epu32(__m128i __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.256
  return _mm256_mask_cvttpd_epu32(__W,__U,__A); 
}
__m128i test_mm256_maskz_cvttpd_epu32(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttpd_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttpd2udq.256
  return _mm256_maskz_cvttpd_epu32(__U,__A); 
}
__m128i test_mm_mask_cvttps_epi32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvttps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.128
  return _mm_mask_cvttps_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttps_epi32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.128
  return _mm_maskz_cvttps_epi32(__U,__A); 
}
__m256i test_mm256_mask_cvttps_epi32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.256
  return _mm256_mask_cvttps_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvttps_epi32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttps_epi32
  // CHECK: @llvm.x86.avx512.mask.cvttps2dq.256
  return _mm256_maskz_cvttps_epi32(__U,__A); 
}
__m128i test_mm_cvttps_epu32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.128
  return _mm_cvttps_epu32(__A); 
}
__m128i test_mm_mask_cvttps_epu32(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.128
  return _mm_mask_cvttps_epu32(__W,__U,__A); 
}
__m128i test_mm_maskz_cvttps_epu32(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.128
  return _mm_maskz_cvttps_epu32(__U,__A); 
}
__m256i test_mm256_cvttps_epu32(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.256
  return _mm256_cvttps_epu32(__A); 
}
__m256i test_mm256_mask_cvttps_epu32(__m256i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.256
  return _mm256_mask_cvttps_epu32(__W,__U,__A); 
}
__m256i test_mm256_maskz_cvttps_epu32(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvttps_epu32
  // CHECK: @llvm.x86.avx512.mask.cvttps2udq.256
  return _mm256_maskz_cvttps_epu32(__U,__A); 
}
__m128d test_mm_cvtepu32_pd(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepu32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtudq2pd.128
  return _mm_cvtepu32_pd(__A); 
}
__m128d test_mm_mask_cvtepu32_pd(__m128d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtudq2pd.128
  return _mm_mask_cvtepu32_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_cvtepu32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtudq2pd.128
  return _mm_maskz_cvtepu32_pd(__U,__A); 
}
__m256d test_mm256_cvtepu32_pd(__m128i __A) {
  // CHECK-LABEL: @test_mm256_cvtepu32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtudq2pd.256
  return _mm256_cvtepu32_pd(__A); 
}
__m256d test_mm256_mask_cvtepu32_pd(__m256d __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtudq2pd.256
  return _mm256_mask_cvtepu32_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_cvtepu32_pd(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu32_pd
  // CHECK: @llvm.x86.avx512.mask.cvtudq2pd.256
  return _mm256_maskz_cvtepu32_pd(__U,__A); 
}
__m128 test_mm_cvtepu32_ps(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepu32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtudq2ps.128
  return _mm_cvtepu32_ps(__A); 
}
__m128 test_mm_mask_cvtepu32_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtudq2ps.128
  return _mm_mask_cvtepu32_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_cvtepu32_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtudq2ps.128
  return _mm_maskz_cvtepu32_ps(__U,__A); 
}
__m256 test_mm256_cvtepu32_ps(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepu32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtudq2ps.256
  return _mm256_cvtepu32_ps(__A); 
}
__m256 test_mm256_mask_cvtepu32_ps(__m256 __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtudq2ps.256
  return _mm256_mask_cvtepu32_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_cvtepu32_ps(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu32_ps
  // CHECK: @llvm.x86.avx512.mask.cvtudq2ps.256
  return _mm256_maskz_cvtepu32_ps(__U,__A); 
}
__m128d test_mm_mask_div_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_div_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.128
  return _mm_mask_div_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_div_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_div_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.128
  return _mm_maskz_div_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_div_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_div_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.256
  return _mm256_mask_div_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_div_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_div_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.256
  return _mm256_maskz_div_pd(__U,__A,__B); 
}
__m128 test_mm_mask_div_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_div_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.128
  return _mm_mask_div_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_div_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_div_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.128
  return _mm_maskz_div_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_div_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_div_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.256
  return _mm256_mask_div_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_div_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_div_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.256
  return _mm256_maskz_div_ps(__U,__A,__B); 
}
__m128d test_mm_mask_expand_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand.pd.128
  return _mm_mask_expand_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_expand_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand.pd.128
  return _mm_maskz_expand_pd(__U,__A); 
}
__m256d test_mm256_mask_expand_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand.pd.256
  return _mm256_mask_expand_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_expand_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_pd
  // CHECK: @llvm.x86.avx512.mask.expand.pd.256
  return _mm256_maskz_expand_pd(__U,__A); 
}
__m128i test_mm_mask_expand_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.q.128
  return _mm_mask_expand_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_expand_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.q.128
  return _mm_maskz_expand_epi64(__U,__A); 
}
__m256i test_mm256_mask_expand_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.q.256
  return _mm256_mask_expand_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_expand_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.q.256
  return _mm256_maskz_expand_epi64(__U,__A); 
}
__m128d test_mm_mask_expandloadu_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_pd
  // CHECK: @llvm.x86.avx512.mask.expand.load.pd.128
  return _mm_mask_expandloadu_pd(__W,__U,__P); 
}
__m128d test_mm_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_pd
  // CHECK: @llvm.x86.avx512.mask.expand.load.pd.128
  return _mm_maskz_expandloadu_pd(__U,__P); 
}
__m256d test_mm256_mask_expandloadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_pd
  // CHECK: @llvm.x86.avx512.mask.expand.load.pd.256
  return _mm256_mask_expandloadu_pd(__W,__U,__P); 
}
__m256d test_mm256_maskz_expandloadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_pd
  // CHECK: @llvm.x86.avx512.mask.expand.load.pd.256
  return _mm256_maskz_expandloadu_pd(__U,__P); 
}
__m128i test_mm_mask_expandloadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.load.q.128
  return _mm_mask_expandloadu_epi64(__W,__U,__P); 
}
__m128i test_mm_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.load.q.128
  return _mm_maskz_expandloadu_epi64(__U,__P); 
}
__m256i test_mm256_mask_expandloadu_epi64(__m256i __W, __mmask8 __U,   void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.load.q.256
  return _mm256_mask_expandloadu_epi64(__W,__U,__P); 
}
__m256i test_mm256_maskz_expandloadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_epi64
  // CHECK: @llvm.x86.avx512.mask.expand.load.q.256
  return _mm256_maskz_expandloadu_epi64(__U,__P); 
}
__m128 test_mm_mask_expandloadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_ps
  // CHECK: @llvm.x86.avx512.mask.expand.load.ps.128
  return _mm_mask_expandloadu_ps(__W,__U,__P); 
}
__m128 test_mm_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_ps
  // CHECK: @llvm.x86.avx512.mask.expand.load.ps.128
  return _mm_maskz_expandloadu_ps(__U,__P); 
}
__m256 test_mm256_mask_expandloadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_ps
  // CHECK: @llvm.x86.avx512.mask.expand.load.ps.256
  return _mm256_mask_expandloadu_ps(__W,__U,__P); 
}
__m256 test_mm256_maskz_expandloadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_ps
  // CHECK: @llvm.x86.avx512.mask.expand.load.ps.256
  return _mm256_maskz_expandloadu_ps(__U,__P); 
}
__m128i test_mm_mask_expandloadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_expandloadu_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.load.d.128
  return _mm_mask_expandloadu_epi32(__W,__U,__P); 
}
__m128i test_mm_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_expandloadu_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.load.d.128
  return _mm_maskz_expandloadu_epi32(__U,__P); 
}
__m256i test_mm256_mask_expandloadu_epi32(__m256i __W, __mmask8 __U,   void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_expandloadu_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.load.d.256
  return _mm256_mask_expandloadu_epi32(__W,__U,__P); 
}
__m256i test_mm256_maskz_expandloadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_expandloadu_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.load.d.256
  return _mm256_maskz_expandloadu_epi32(__U,__P); 
}
__m128 test_mm_mask_expand_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand.ps.128
  return _mm_mask_expand_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_expand_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand.ps.128
  return _mm_maskz_expand_ps(__U,__A); 
}
__m256 test_mm256_mask_expand_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand.ps.256
  return _mm256_mask_expand_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_expand_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_ps
  // CHECK: @llvm.x86.avx512.mask.expand.ps.256
  return _mm256_maskz_expand_ps(__U,__A); 
}
__m128i test_mm_mask_expand_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.d.128
  return _mm_mask_expand_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_expand_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.d.128
  return _mm_maskz_expand_epi32(__U,__A); 
}
__m256i test_mm256_mask_expand_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.d.256
  return _mm256_mask_expand_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_expand_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_expand_epi32
  // CHECK: @llvm.x86.avx512.mask.expand.d.256
  return _mm256_maskz_expand_epi32(__U,__A); 
}
__m128d test_mm_getexp_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.128
  return _mm_getexp_pd(__A); 
}
__m128d test_mm_mask_getexp_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.128
  return _mm_mask_getexp_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_getexp_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.128
  return _mm_maskz_getexp_pd(__U,__A); 
}
__m256d test_mm256_getexp_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.256
  return _mm256_getexp_pd(__A); 
}
__m256d test_mm256_mask_getexp_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.256
  return _mm256_mask_getexp_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_getexp_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.256
  return _mm256_maskz_getexp_pd(__U,__A); 
}
__m128 test_mm_getexp_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.128
  return _mm_getexp_ps(__A); 
}
__m128 test_mm_mask_getexp_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.128
  return _mm_mask_getexp_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_getexp_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.128
  return _mm_maskz_getexp_ps(__U,__A); 
}
__m256 test_mm256_getexp_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.256
  return _mm256_getexp_ps(__A); 
}
__m256 test_mm256_mask_getexp_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.256
  return _mm256_mask_getexp_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_getexp_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.256
  return _mm256_maskz_getexp_ps(__U,__A); 
}
__m128d test_mm_mask_max_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_max_pd
  // CHECK: @llvm.x86.avx512.mask.max.pd
  return _mm_mask_max_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_max_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_max_pd
  // CHECK: @llvm.x86.avx512.mask.max.pd
  return _mm_maskz_max_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_max_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_max_pd
  // CHECK: @llvm.x86.avx512.mask.max.pd.256
  return _mm256_mask_max_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_max_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_pd
  // CHECK: @llvm.x86.avx512.mask.max.pd.256
  return _mm256_maskz_max_pd(__U,__A,__B); 
}
__m128 test_mm_mask_max_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_max_ps
  // CHECK: @llvm.x86.avx512.mask.max.ps
  return _mm_mask_max_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_max_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_max_ps
  // CHECK: @llvm.x86.avx512.mask.max.ps
  return _mm_maskz_max_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_max_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_max_ps
  // CHECK: @llvm.x86.avx512.mask.max.ps.256
  return _mm256_mask_max_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_max_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_ps
  // CHECK: @llvm.x86.avx512.mask.max.ps.256
  return _mm256_maskz_max_ps(__U,__A,__B); 
}
__m128d test_mm_mask_min_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_min_pd
  // CHECK: @llvm.x86.avx512.mask.min.pd
  return _mm_mask_min_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_min_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_min_pd
  // CHECK: @llvm.x86.avx512.mask.min.pd
  return _mm_maskz_min_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_min_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_min_pd
  // CHECK: @llvm.x86.avx512.mask.min.pd.256
  return _mm256_mask_min_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_min_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_pd
  // CHECK: @llvm.x86.avx512.mask.min.pd.256
  return _mm256_maskz_min_pd(__U,__A,__B); 
}
__m128 test_mm_mask_min_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_min_ps
  // CHECK: @llvm.x86.avx512.mask.min.ps
  return _mm_mask_min_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_min_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_min_ps
  // CHECK: @llvm.x86.avx512.mask.min.ps
  return _mm_maskz_min_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_min_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_min_ps
  // CHECK: @llvm.x86.avx512.mask.min.ps.256
  return _mm256_mask_min_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_min_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_ps
  // CHECK: @llvm.x86.avx512.mask.min.ps.256
  return _mm256_maskz_min_ps(__U,__A,__B); 
}
__m128d test_mm_mask_mul_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_mul_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd
  return _mm_mask_mul_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_mul_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd
  return _mm_maskz_mul_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_mul_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_mul_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.256
  return _mm256_mask_mul_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_mul_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_mul_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.256
  return _mm256_maskz_mul_pd(__U,__A,__B); 
}
__m128 test_mm_mask_mul_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_mul_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps
  return _mm_mask_mul_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_mul_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps
  return _mm_maskz_mul_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_mul_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_mul_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.256
  return _mm256_mask_mul_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_mul_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_mul_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.256
  return _mm256_maskz_mul_ps(__U,__A,__B); 
}
__m128i test_mm_mask_abs_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi32
  // CHECK: @llvm.x86.avx512.mask.pabs.d.128
  return _mm_mask_abs_epi32(__W,__U,__A); 
}
__m128i test_mm_maskz_abs_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi32
  // CHECK: @llvm.x86.avx512.mask.pabs.d.128
  return _mm_maskz_abs_epi32(__U,__A); 
}
__m256i test_mm256_mask_abs_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi32
  // CHECK: @llvm.x86.avx512.mask.pabs.d.256
  return _mm256_mask_abs_epi32(__W,__U,__A); 
}
__m256i test_mm256_maskz_abs_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi32
  // CHECK: @llvm.x86.avx512.mask.pabs.d.256
  return _mm256_maskz_abs_epi32(__U,__A); 
}
__m128i test_mm_abs_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_abs_epi64
  // CHECK: @llvm.x86.avx512.mask.pabs.q.128
  return _mm_abs_epi64(__A); 
}
__m128i test_mm_mask_abs_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi64
  // CHECK: @llvm.x86.avx512.mask.pabs.q.128
  return _mm_mask_abs_epi64(__W,__U,__A); 
}
__m128i test_mm_maskz_abs_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi64
  // CHECK: @llvm.x86.avx512.mask.pabs.q.128
  return _mm_maskz_abs_epi64(__U,__A); 
}
__m256i test_mm256_abs_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_abs_epi64
  // CHECK: @llvm.x86.avx512.mask.pabs.q.256
  return _mm256_abs_epi64(__A); 
}
__m256i test_mm256_mask_abs_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi64
  // CHECK: @llvm.x86.avx512.mask.pabs.q.256
  return _mm256_mask_abs_epi64(__W,__U,__A); 
}
__m256i test_mm256_maskz_abs_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi64
  // CHECK: @llvm.x86.avx512.mask.pabs.q.256
  return _mm256_maskz_abs_epi64(__U,__A); 
}
__m128i test_mm_maskz_max_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi32
  // CHECK: @llvm.x86.avx512.mask.pmaxs.d.128
  return _mm_maskz_max_epi32(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi32
  // CHECK: @llvm.x86.avx512.mask.pmaxs.d.128
  return _mm_mask_max_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epi32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi32
  // CHECK: @llvm.x86.avx512.mask.pmaxs.d.256
  return _mm256_maskz_max_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi32
  // CHECK: @llvm.x86.avx512.mask.pmaxs.d.256
  return _mm256_mask_max_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epi64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi64
  // CHECK: @llvm.x86.avx512.mask.pmaxs.q.128
  return _mm_maskz_max_epi64(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi64
  // CHECK: @llvm.x86.avx512.mask.pmaxs.q.128
  return _mm_mask_max_epi64(__W,__M,__A,__B); 
}
__m128i test_mm_max_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_max_epi64
  // CHECK: @llvm.x86.avx512.mask.pmaxs.q.128
  return _mm_max_epi64(__A,__B); 
}
__m256i test_mm256_maskz_max_epi64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi64
  // CHECK: @llvm.x86.avx512.mask.pmaxs.q.256
  return _mm256_maskz_max_epi64(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi64
  // CHECK: @llvm.x86.avx512.mask.pmaxs.q.256
  return _mm256_mask_max_epi64(__W,__M,__A,__B); 
}
__m256i test_mm256_max_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_max_epi64
  // CHECK: @llvm.x86.avx512.mask.pmaxs.q.256
  return _mm256_max_epi64(__A,__B); 
}
__m128i test_mm_maskz_max_epu32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu32
  // CHECK: @llvm.x86.avx512.mask.pmaxu.d.128
  return _mm_maskz_max_epu32(__M,__A,__B); 
}
__m128i test_mm_mask_max_epu32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu32
  // CHECK: @llvm.x86.avx512.mask.pmaxu.d.128
  return _mm_mask_max_epu32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu32
  // CHECK: @llvm.x86.avx512.mask.pmaxu.d.256
  return _mm256_maskz_max_epu32(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epu32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu32
  // CHECK: @llvm.x86.avx512.mask.pmaxu.d.256
  return _mm256_mask_max_epu32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epu64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu64
  // CHECK: @llvm.x86.avx512.mask.pmaxu.q.128
  return _mm_maskz_max_epu64(__M,__A,__B); 
}
__m128i test_mm_max_epu64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_max_epu64
  // CHECK: @llvm.x86.avx512.mask.pmaxu.q.128
  return _mm_max_epu64(__A,__B); 
}
__m128i test_mm_mask_max_epu64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu64
  // CHECK: @llvm.x86.avx512.mask.pmaxu.q.128
  return _mm_mask_max_epu64(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu64
  // CHECK: @llvm.x86.avx512.mask.pmaxu.q.256
  return _mm256_maskz_max_epu64(__M,__A,__B); 
}
__m256i test_mm256_max_epu64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_max_epu64
  // CHECK: @llvm.x86.avx512.mask.pmaxu.q.256
  return _mm256_max_epu64(__A,__B); 
}
__m256i test_mm256_mask_max_epu64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu64
  // CHECK: @llvm.x86.avx512.mask.pmaxu.q.256
  return _mm256_mask_max_epu64(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi32
  // CHECK: @llvm.x86.avx512.mask.pmins.d.128
  return _mm_maskz_min_epi32(__M,__A,__B); 
}
__m128i test_mm_mask_min_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi32
  // CHECK: @llvm.x86.avx512.mask.pmins.d.128
  return _mm_mask_min_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi32
  // CHECK: @llvm.x86.avx512.mask.pmins.d.256
  return _mm256_maskz_min_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epi32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi32
  // CHECK: @llvm.x86.avx512.mask.pmins.d.256
  return _mm256_mask_min_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_min_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_min_epi64
  // CHECK: @llvm.x86.avx512.mask.pmins.q.128
  return _mm_min_epi64(__A,__B); 
}
__m128i test_mm_mask_min_epi64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi64
  // CHECK: @llvm.x86.avx512.mask.pmins.q.128
  return _mm_mask_min_epi64(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi64
  // CHECK: @llvm.x86.avx512.mask.pmins.q.128
  return _mm_maskz_min_epi64(__M,__A,__B); 
}
__m256i test_mm256_min_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_min_epi64
  // CHECK: @llvm.x86.avx512.mask.pmins.q.256
  return _mm256_min_epi64(__A,__B); 
}
__m256i test_mm256_mask_min_epi64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi64
  // CHECK: @llvm.x86.avx512.mask.pmins.q.256
  return _mm256_mask_min_epi64(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi64
  // CHECK: @llvm.x86.avx512.mask.pmins.q.256
  return _mm256_maskz_min_epi64(__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu32
  // CHECK: @llvm.x86.avx512.mask.pminu.d.128
  return _mm_maskz_min_epu32(__M,__A,__B); 
}
__m128i test_mm_mask_min_epu32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu32
  // CHECK: @llvm.x86.avx512.mask.pminu.d.128
  return _mm_mask_min_epu32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu32(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu32
  // CHECK: @llvm.x86.avx512.mask.pminu.d.256
  return _mm256_maskz_min_epu32(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epu32(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu32
  // CHECK: @llvm.x86.avx512.mask.pminu.d.256
  return _mm256_mask_min_epu32(__W,__M,__A,__B); 
}
__m128i test_mm_min_epu64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_min_epu64
  // CHECK: @llvm.x86.avx512.mask.pminu.q.128
  return _mm_min_epu64(__A,__B); 
}
__m128i test_mm_mask_min_epu64(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu64
  // CHECK: @llvm.x86.avx512.mask.pminu.q.128
  return _mm_mask_min_epu64(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu64(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu64
  // CHECK: @llvm.x86.avx512.mask.pminu.q.128
  return _mm_maskz_min_epu64(__M,__A,__B); 
}
__m256i test_mm256_min_epu64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_min_epu64
  // CHECK: @llvm.x86.avx512.mask.pminu.q.256
  return _mm256_min_epu64(__A,__B); 
}
__m256i test_mm256_mask_min_epu64(__m256i __W, __mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu64
  // CHECK: @llvm.x86.avx512.mask.pminu.q.256
  return _mm256_mask_min_epu64(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu64(__mmask8 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu64
  // CHECK: @llvm.x86.avx512.mask.pminu.q.256
  return _mm256_maskz_min_epu64(__M,__A,__B); 
}
__m128d test_mm_roundscale_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.128
  return _mm_roundscale_pd(__A,4); 
}
__m128d test_mm_mask_roundscale_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.128
  return _mm_mask_roundscale_pd(__W,__U,__A,4); 
}
__m128d test_mm_maskz_roundscale_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.128
  return _mm_maskz_roundscale_pd(__U,__A,4); 
}
__m256d test_mm256_roundscale_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.256
  return _mm256_roundscale_pd(__A,4); 
}
__m256d test_mm256_mask_roundscale_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.256
  return _mm256_mask_roundscale_pd(__W,__U,__A,4); 
}
__m256d test_mm256_maskz_roundscale_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_roundscale_pd
  // CHECK: @llvm.x86.avx512.mask.rndscale.pd.256
  return _mm256_maskz_roundscale_pd(__U,__A,4); 
}
__m128 test_mm_roundscale_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.128
  return _mm_roundscale_ps(__A,4); 
}
__m128 test_mm_mask_roundscale_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.128
  return _mm_mask_roundscale_ps(__W,__U,__A,4); 
}
__m128 test_mm_maskz_roundscale_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.128
  return _mm_maskz_roundscale_ps(__U,__A, 4); 
}
__m256 test_mm256_roundscale_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.256
  return _mm256_roundscale_ps(__A,4); 
}
__m256 test_mm256_mask_roundscale_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.256
  return _mm256_mask_roundscale_ps(__W,__U,__A,4); 
}
__m256 test_mm256_maskz_roundscale_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_roundscale_ps
  // CHECK: @llvm.x86.avx512.mask.rndscale.ps.256
  return _mm256_maskz_roundscale_ps(__U,__A,4); 
}
__m128d test_mm_scalef_pd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.128
  return _mm_scalef_pd(__A,__B); 
}
__m128d test_mm_mask_scalef_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.128
  return _mm_mask_scalef_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_scalef_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.128
  return _mm_maskz_scalef_pd(__U,__A,__B); 
}
__m256d test_mm256_scalef_pd(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.256
  return _mm256_scalef_pd(__A,__B); 
}
__m256d test_mm256_mask_scalef_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.256
  return _mm256_mask_scalef_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_scalef_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.256
  return _mm256_maskz_scalef_pd(__U,__A,__B); 
}
__m128 test_mm_scalef_ps(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.128
  return _mm_scalef_ps(__A,__B); 
}
__m128 test_mm_mask_scalef_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.128
  return _mm_mask_scalef_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_scalef_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.128
  return _mm_maskz_scalef_ps(__U,__A,__B); 
}
__m256 test_mm256_scalef_ps(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.256
  return _mm256_scalef_ps(__A,__B); 
}
__m256 test_mm256_mask_scalef_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.256
  return _mm256_mask_scalef_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_scalef_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.256
  return _mm256_maskz_scalef_ps(__U,__A,__B); 
}
void test_mm_i64scatter_pd(double *__addr, __m128i __index,  __m128d __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_pd
  // CHECK: @llvm.x86.avx512.scatterdiv2.df
  return _mm_i64scatter_pd(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m128d __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_pd
  // CHECK: @llvm.x86.avx512.scatterdiv2.df
  return _mm_mask_i64scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm_i64scatter_epi64(long long *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.scatterdiv2.di
  return _mm_i64scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_epi64(long long *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.scatterdiv2.di
  return _mm_mask_i64scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_pd(double *__addr, __m256i __index,  __m256d __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_pd
  // CHECK: @llvm.x86.avx512.scatterdiv4.df
  return _mm256_i64scatter_pd(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_pd(double *__addr, __mmask8 __mask, __m256i __index, __m256d __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_pd
  // CHECK: @llvm.x86.avx512.scatterdiv4.df
  return _mm256_mask_i64scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_epi64(long long *__addr, __m256i __index,  __m256i __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.scatterdiv4.di
  return _mm256_i64scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_epi64(long long *__addr, __mmask8 __mask,  __m256i __index, __m256i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_epi64
  // CHECK: @llvm.x86.avx512.scatterdiv4.di
  return _mm256_mask_i64scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm_i64scatter_ps(float *__addr, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_ps
  // CHECK: @llvm.x86.avx512.scatterdiv4.sf
  return _mm_i64scatter_ps(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_ps(float *__addr, __mmask8 __mask, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_ps
  // CHECK: @llvm.x86.avx512.scatterdiv4.sf
  return _mm_mask_i64scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm_i64scatter_epi32(int *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.scatterdiv4.si
  return _mm_i64scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm_mask_i64scatter_epi32(int *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.scatterdiv4.si
  return _mm_mask_i64scatter_epi32(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_ps(float *__addr, __m256i __index,  __m128 __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_ps
  // CHECK: @llvm.x86.avx512.scatterdiv8.sf
  return _mm256_i64scatter_ps(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_ps(float *__addr, __mmask8 __mask, __m256i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_ps
  // CHECK: @llvm.x86.avx512.scatterdiv8.sf
  return _mm256_mask_i64scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i64scatter_epi32(int *__addr, __m256i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm256_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.scatterdiv8.si
  return _mm256_i64scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm256_mask_i64scatter_epi32(int *__addr, __mmask8 __mask,  __m256i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i64scatter_epi32
  // CHECK: @llvm.x86.avx512.scatterdiv8.si
  return _mm256_mask_i64scatter_epi32(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_pd(double *__addr, __m128i __index,  __m128d __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_pd
  // CHECK: @llvm.x86.avx512.scattersiv2.df
  return _mm_i32scatter_pd(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m128d __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_pd
  // CHECK: @llvm.x86.avx512.scattersiv2.df
  return _mm_mask_i32scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_epi64(long long *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.scattersiv2.di
  return _mm_i32scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_epi64(long long *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.scattersiv2.di
  return _mm_mask_i32scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_pd(double *__addr, __m128i __index,  __m256d __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_pd
  // CHECK: @llvm.x86.avx512.scattersiv4.df
  return _mm256_i32scatter_pd(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_pd(double *__addr, __mmask8 __mask, __m128i __index, __m256d __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_pd
  // CHECK: @llvm.x86.avx512.scattersiv4.df
  return _mm256_mask_i32scatter_pd(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_epi64(long long *__addr, __m128i __index,  __m256i __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.scattersiv4.di
  return _mm256_i32scatter_epi64(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_epi64(long long *__addr, __mmask8 __mask,  __m128i __index, __m256i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_epi64
  // CHECK: @llvm.x86.avx512.scattersiv4.di
  return _mm256_mask_i32scatter_epi64(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_ps(float *__addr, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_ps
  // CHECK: @llvm.x86.avx512.scattersiv4.sf
  return _mm_i32scatter_ps(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_ps(float *__addr, __mmask8 __mask, __m128i __index, __m128 __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_ps
  // CHECK: @llvm.x86.avx512.scattersiv4.sf
  return _mm_mask_i32scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm_i32scatter_epi32(int *__addr, __m128i __index,  __m128i __v1) {
  // CHECK-LABEL: @test_mm_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.scattersiv4.si
  return _mm_i32scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm_mask_i32scatter_epi32(int *__addr, __mmask8 __mask, __m128i __index, __m128i __v1) {
  // CHECK-LABEL: @test_mm_mask_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.scattersiv4.si
  return _mm_mask_i32scatter_epi32(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_ps(float *__addr, __m256i __index,  __m256 __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_ps
  // CHECK: @llvm.x86.avx512.scattersiv8.sf
  return _mm256_i32scatter_ps(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_ps(float *__addr, __mmask8 __mask, __m256i __index, __m256 __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_ps
  // CHECK: @llvm.x86.avx512.scattersiv8.sf
  return _mm256_mask_i32scatter_ps(__addr,__mask,__index,__v1,2); 
}
void test_mm256_i32scatter_epi32(int *__addr, __m256i __index,  __m256i __v1) {
  // CHECK-LABEL: @test_mm256_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.scattersiv8.si
  return _mm256_i32scatter_epi32(__addr,__index,__v1,2); 
}
void test_mm256_mask_i32scatter_epi32(int *__addr, __mmask8 __mask,  __m256i __index, __m256i __v1) {
  // CHECK-LABEL: @test_mm256_mask_i32scatter_epi32
  // CHECK: @llvm.x86.avx512.scattersiv8.si
  return _mm256_mask_i32scatter_epi32(__addr,__mask,__index,__v1,2); 
}
__m128d test_mm_mask_sqrt_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_sqrt_pd
  // CHECK: @llvm.x86.avx512.mask.sqrt.pd.128
  return _mm_mask_sqrt_pd(__W,__U,__A); 
}
__m128d test_mm_maskz_sqrt_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_sqrt_pd
  // CHECK: @llvm.x86.avx512.mask.sqrt.pd.128
  return _mm_maskz_sqrt_pd(__U,__A); 
}
__m256d test_mm256_mask_sqrt_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_sqrt_pd
  // CHECK: @llvm.x86.avx512.mask.sqrt.pd.256
  return _mm256_mask_sqrt_pd(__W,__U,__A); 
}
__m256d test_mm256_maskz_sqrt_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_sqrt_pd
  // CHECK: @llvm.x86.avx512.mask.sqrt.pd.256
  return _mm256_maskz_sqrt_pd(__U,__A); 
}
__m128 test_mm_mask_sqrt_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_sqrt_ps
  // CHECK: @llvm.x86.avx512.mask.sqrt.ps.128
  return _mm_mask_sqrt_ps(__W,__U,__A); 
}
__m128 test_mm_maskz_sqrt_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_sqrt_ps
  // CHECK: @llvm.x86.avx512.mask.sqrt.ps.128
  return _mm_maskz_sqrt_ps(__U,__A); 
}
__m256 test_mm256_mask_sqrt_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_sqrt_ps
  // CHECK: @llvm.x86.avx512.mask.sqrt.ps.256
  return _mm256_mask_sqrt_ps(__W,__U,__A); 
}
__m256 test_mm256_maskz_sqrt_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_sqrt_ps
  // CHECK: @llvm.x86.avx512.mask.sqrt.ps.256
  return _mm256_maskz_sqrt_ps(__U,__A); 
}
__m128d test_mm_mask_sub_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_sub_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.128
  return _mm_mask_sub_pd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_sub_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.128
  return _mm_maskz_sub_pd(__U,__A,__B); 
}
__m256d test_mm256_mask_sub_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_sub_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.256
  return _mm256_mask_sub_pd(__W,__U,__A,__B); 
}
__m256d test_mm256_maskz_sub_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_sub_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.256
  return _mm256_maskz_sub_pd(__U,__A,__B); 
}
__m128 test_mm_mask_sub_ps(__m128 __W, __mmask16 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_sub_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.128
  return _mm_mask_sub_ps(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_sub_ps(__mmask16 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.128
  return _mm_maskz_sub_ps(__U,__A,__B); 
}
__m256 test_mm256_mask_sub_ps(__m256 __W, __mmask16 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_sub_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.256
  return _mm256_mask_sub_ps(__W,__U,__A,__B); 
}
__m256 test_mm256_maskz_sub_ps(__mmask16 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_sub_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.256
  return _mm256_maskz_sub_ps(__U,__A,__B); 
}
__m128i test_mm_mask2_permutex2var_epi32(__m128i __A, __m128i __I, __mmask8 __U,  __m128i __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.d.128
  return _mm_mask2_permutex2var_epi32(__A,__I,__U,__B); 
}
__m256i test_mm256_mask2_permutex2var_epi32(__m256i __A, __m256i __I, __mmask8 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.d.256
  return _mm256_mask2_permutex2var_epi32(__A,__I,__U,__B); 
}
__m128d test_mm_mask2_permutex2var_pd(__m128d __A, __m128i __I, __mmask8 __U, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.pd.128
  return _mm_mask2_permutex2var_pd(__A,__I,__U,__B); 
}
__m256d test_mm256_mask2_permutex2var_pd(__m256d __A, __m256i __I, __mmask8 __U,  __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.pd.256
  return _mm256_mask2_permutex2var_pd(__A,__I,__U,__B); 
}
__m128 test_mm_mask2_permutex2var_ps(__m128 __A, __m128i __I, __mmask8 __U, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.ps.128
  return _mm_mask2_permutex2var_ps(__A,__I,__U,__B); 
}
__m256 test_mm256_mask2_permutex2var_ps(__m256 __A, __m256i __I, __mmask8 __U,  __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.ps.256
  return _mm256_mask2_permutex2var_ps(__A,__I,__U,__B); 
}
__m128i test_mm_mask2_permutex2var_epi64(__m128i __A, __m128i __I, __mmask8 __U,  __m128i __B) {
  // CHECK-LABEL: @test_mm_mask2_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.q.128
  return _mm_mask2_permutex2var_epi64(__A,__I,__U,__B); 
}
__m256i test_mm256_mask2_permutex2var_epi64(__m256i __A, __m256i __I, __mmask8 __U, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask2_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.q.256
  return _mm256_mask2_permutex2var_epi64(__A,__I,__U,__B); 
}
__m128i test_mm_permutex2var_epi32(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.d.128
  return _mm_permutex2var_epi32(__A,__I,__B); 
}
__m128i test_mm_mask_permutex2var_epi32(__m128i __A, __mmask8 __U, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.d.128
  return _mm_mask_permutex2var_epi32(__A,__U,__I,__B); 
}
__m128i test_mm_maskz_permutex2var_epi32(__mmask8 __U, __m128i __A, __m128i __I,  __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.d.128
  return _mm_maskz_permutex2var_epi32(__U,__A,__I,__B); 
}
__m256i test_mm256_permutex2var_epi32(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.d.256
  return _mm256_permutex2var_epi32(__A,__I,__B); 
}
__m256i test_mm256_mask_permutex2var_epi32(__m256i __A, __mmask8 __U, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.d.256
  return _mm256_mask_permutex2var_epi32(__A,__U,__I,__B); 
}
__m256i test_mm256_maskz_permutex2var_epi32(__mmask8 __U, __m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.d.256
  return _mm256_maskz_permutex2var_epi32(__U,__A,__I,__B); 
}
__m128d test_mm_permutex2var_pd(__m128d __A, __m128i __I, __m128d __B) {
  // CHECK-LABEL: @test_mm_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.pd.128
  return _mm_permutex2var_pd(__A,__I,__B); 
}
__m128d test_mm_mask_permutex2var_pd(__m128d __A, __mmask8 __U, __m128i __I, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.pd.128
  return _mm_mask_permutex2var_pd(__A,__U,__I,__B); 
}
__m128d test_mm_maskz_permutex2var_pd(__mmask8 __U, __m128d __A, __m128i __I, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_pd
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.pd.128
  return _mm_maskz_permutex2var_pd(__U,__A,__I,__B); 
}
__m256d test_mm256_permutex2var_pd(__m256d __A, __m256i __I, __m256d __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.pd.256
  return _mm256_permutex2var_pd(__A,__I,__B); 
}
__m256d test_mm256_mask_permutex2var_pd(__m256d __A, __mmask8 __U, __m256i __I, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.pd.256
  return _mm256_mask_permutex2var_pd(__A,__U,__I,__B); 
}
__m256d test_mm256_maskz_permutex2var_pd(__mmask8 __U, __m256d __A, __m256i __I,  __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_pd
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.pd.256
  return _mm256_maskz_permutex2var_pd(__U,__A,__I,__B); 
}
__m128 test_mm_permutex2var_ps(__m128 __A, __m128i __I, __m128 __B) {
  // CHECK-LABEL: @test_mm_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.ps.128
  return _mm_permutex2var_ps(__A,__I,__B); 
}
__m128 test_mm_mask_permutex2var_ps(__m128 __A, __mmask8 __U, __m128i __I, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.ps.128
  return _mm_mask_permutex2var_ps(__A,__U,__I,__B); 
}
__m128 test_mm_maskz_permutex2var_ps(__mmask8 __U, __m128 __A, __m128i __I, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_ps
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.ps.128
  return _mm_maskz_permutex2var_ps(__U,__A,__I,__B); 
}
__m256 test_mm256_permutex2var_ps(__m256 __A, __m256i __I, __m256 __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.ps.256
  return _mm256_permutex2var_ps(__A,__I,__B); 
}
__m256 test_mm256_mask_permutex2var_ps(__m256 __A, __mmask8 __U, __m256i __I, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.ps.256
  return _mm256_mask_permutex2var_ps(__A,__U,__I,__B); 
}
__m256 test_mm256_maskz_permutex2var_ps(__mmask8 __U, __m256 __A, __m256i __I, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_ps
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.ps.256
  return _mm256_maskz_permutex2var_ps(__U,__A,__I,__B); 
}
__m128i test_mm_permutex2var_epi64(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.q.128
  return _mm_permutex2var_epi64(__A,__I,__B); 
}
__m128i test_mm_mask_permutex2var_epi64(__m128i __A, __mmask8 __U, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.q.128
  return _mm_mask_permutex2var_epi64(__A,__U,__I,__B); 
}
__m128i test_mm_maskz_permutex2var_epi64(__mmask8 __U, __m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.q.128
  return _mm_maskz_permutex2var_epi64(__U,__A,__I,__B); 
}
__m256i test_mm256_permutex2var_epi64(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.q.256
  return _mm256_permutex2var_epi64(__A,__I,__B); 
}
__m256i test_mm256_mask_permutex2var_epi64(__m256i __A, __mmask8 __U, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermt2var.q.256
  return _mm256_mask_permutex2var_epi64(__A,__U,__I,__B); 
}
__m256i test_mm256_maskz_permutex2var_epi64(__mmask8 __U, __m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.q.256
  return _mm256_maskz_permutex2var_epi64(__U,__A,__I,__B); 
}
