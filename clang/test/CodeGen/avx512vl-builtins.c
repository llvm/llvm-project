// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__mmask8 test_mm_cmpeq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  return (__mmask8)_mm_cmpeq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <2 x i1> %{{.*}}, <2 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3>
  return (__mmask8)_mm_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi32_mask
  // CHECK: icmp sge <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi32_mask
  // CHECK: icmp sge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi64_mask
  // CHECK: icmp sge <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi64_mask
  // CHECK: icmp sge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi32_mask
  // CHECK: icmp sge <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi32_mask
  // CHECK: icmp sge <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi64_mask
  // CHECK: icmp sge <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi64_mask
  // CHECK: icmp sge <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu32_mask
  // CHECK: icmp uge <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu32_mask
  // CHECK: icmp uge <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu64_mask
  // CHECK: icmp uge <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu64_mask
  // CHECK: icmp uge <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu32_mask
  // CHECK: icmp uge <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu32_mask
  // CHECK: icmp uge <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpge_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu64_mask
  // CHECK: icmp uge <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpge_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu64_mask
  // CHECK: icmp uge <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu32_mask
  // CHECK: icmp ugt <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu32_mask
  // CHECK: icmp ugt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu64_mask
  // CHECK: icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu64_mask
  // CHECK: icmp ugt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu32_mask
  // CHECK: icmp ugt <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu32_mask
  // CHECK: icmp ugt <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu64_mask
  // CHECK: icmp ugt <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu64_mask
  // CHECK: icmp ugt <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi32_mask
  // CHECK: icmp sle <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi32_mask
  // CHECK: icmp sle <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi64_mask
  // CHECK: icmp sle <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi64_mask
  // CHECK: icmp sle <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi32_mask
  // CHECK: icmp sle <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi32_mask
  // CHECK: icmp sle <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi64_mask
  // CHECK: icmp sle <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi64_mask
  // CHECK: icmp sle <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu32_mask
  // CHECK: icmp ule <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu32_mask
  // CHECK: icmp ule <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu64_mask
  // CHECK: icmp ule <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu64_mask
  // CHECK: icmp ule <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu32_mask
  // CHECK: icmp ule <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu32_mask
  // CHECK: icmp ule <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmple_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu64_mask
  // CHECK: icmp ule <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmple_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu64_mask
  // CHECK: icmp ule <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi32_mask
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi32_mask
  // CHECK: icmp slt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi64_mask
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi64_mask
  // CHECK: icmp slt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi32_mask
  // CHECK: icmp slt <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi32_mask
  // CHECK: icmp slt <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi64_mask
  // CHECK: icmp slt <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi64_mask
  // CHECK: icmp slt <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu32_mask
  // CHECK: icmp ult <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu32_mask
  // CHECK: icmp ult <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu64_mask
  // CHECK: icmp ult <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu64_mask
  // CHECK: icmp ult <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu32_mask
  // CHECK: icmp ult <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu32_mask
  // CHECK: icmp ult <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmplt_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu64_mask
  // CHECK: icmp ult <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmplt_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu64_mask
  // CHECK: icmp ult <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu32_mask
  // CHECK: icmp ne <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu64_mask
  // CHECK: icmp ne <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epu32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu32_mask
  // CHECK: icmp ne <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpneq_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpneq_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu64_mask
  // CHECK: icmp ne <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmp_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi32_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi64_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi64_mask(__u, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epi32_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epi32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epi64_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epi64_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu32_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu64_mask(__u, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epu32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epu32_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epu32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epu32_mask(__u, __a, __b, 0);
}

__mmask8 test_mm256_cmp_epu64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmp_epu64_mask(__a, __b, 0);
}

__mmask8 test_mm256_mask_cmp_epu64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmp_epu64_mask(__u, __a, __b, 0);
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
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mask_and_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi32
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_and_epi32(__U, __A, __B);
}

__m128i test_mm_mask_and_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi32
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mask_and_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_and_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi32
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_and_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_andnot_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi32
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_andnot_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi32
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_andnot_epi32(__U, __A, __B);
}

__m128i test_mm_mask_andnot_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi32
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_andnot_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi32
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_andnot_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_or_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi32
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mask_or_epi32(__W, __U, __A, __B);
}

 __m256i test_mm256_maskz_or_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi32
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_or_epi32(__U, __A, __B);
}

 __m128i test_mm_mask_or_epi32 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi32
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mask_or_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_or_epi32
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_or_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_xor_epi32 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi32
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mask_xor_epi32(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi32 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi32
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_xor_epi32(__U, __A, __B);
}

__m128i test_mm_mask_xor_epi32 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi32
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mask_xor_epi32(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi32 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi32
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_xor_epi32(__U, __A, __B);
}

__m256i test_mm256_mask_and_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
           __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_and_epi64
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mask_and_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_and_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_and_epi64
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_and_epi64(__U, __A, __B);
}

__m128i test_mm_mask_and_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
        __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_and_epi64
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mask_and_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_and_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_and_epi64
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_and_epi64(__U, __A, __B);
}

__m256i test_mm256_mask_andnot_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
        __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_andnot_epi64
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_andnot_epi64(__W, __U, __A, __B);
}

__m256i test_mm256_maskz_andnot_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_andnot_epi64
  //CHECK: xor <4 x i64> %{{.*}}, <i64 -1, i64 -1, i64 -1, i64 -1>
  //CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_andnot_epi64(__U, __A, __B);
}

__m128i test_mm_mask_andnot_epi64 (__m128i __W, __mmask8 __U, __m128i __A,
           __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_andnot_epi64
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_andnot_epi64(__W,__U, __A, __B);
}

__m128i test_mm_maskz_andnot_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_andnot_epi64
  //CHECK: xor <2 x i64> %{{.*}}, <i64 -1, i64 -1>
  //CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_andnot_epi64(__U, __A, __B);
}

__m256i test_mm256_mask_or_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_or_epi64
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mask_or_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_or_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_or_epi64
  //CHECK: or <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_or_epi64(__U, __A, __B);
}

__m128i test_mm_mask_or_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_or_epi64
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mask_or_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_or_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_or_epi64
  //CHECK: or <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_or_epi64( __U, __A, __B);
}

__m256i test_mm256_mask_xor_epi64 (__m256i __W, __mmask8 __U, __m256i __A,
          __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_xor_epi64
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  return _mm256_mask_xor_epi64(__W,__U, __A, __B);
}

__m256i test_mm256_maskz_xor_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_xor_epi64
  //CHECK: xor <4 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_xor_epi64(__U, __A, __B);
}

__m128i test_mm_mask_xor_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_xor_epi64
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  return _mm_mask_xor_epi64(__W, __U, __A, __B);
}

__m128i test_mm_maskz_xor_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_xor_epi64
  //CHECK: xor <2 x i64> %{{.*}}, %{{.*}}
  //CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
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

__mmask8 test_mm_cmp_ps_mask(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.128
  return (__mmask8)_mm_cmp_ps_mask(__A, __B, 0);
}

__mmask8 test_mm_mask_cmp_ps_mask(__mmask8 m, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.128
  return _mm_mask_cmp_ps_mask(m, __A, __B, 0);
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

__mmask8 test_mm_cmp_pd_mask(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.128
  return (__mmask8)_mm_cmp_pd_mask(__A, __B, 0);
}

__mmask8 test_mm_mask_cmp_pd_mask(__mmask8 m, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.128
  return _mm_mask_cmp_pd_mask(m, __A, __B, 0);
}

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
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_blend_epi32(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi32(__mmask8 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_blend_epi32(__U,__A,__W); 
}
__m128d test_mm_mask_blend_pd(__mmask8 __U, __m128d __A, __m128d __W) {
  // CHECK-LABEL: @test_mm_mask_blend_pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_blend_pd(__U,__A,__W); 
}
__m256d test_mm256_mask_blend_pd(__mmask8 __U, __m256d __A, __m256d __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_pd
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_blend_pd(__U,__A,__W); 
}
__m128 test_mm_mask_blend_ps(__mmask8 __U, __m128 __A, __m128 __W) {
  // CHECK-LABEL: @test_mm_mask_blend_ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_blend_ps(__U,__A,__W); 
}
__m256 test_mm256_mask_blend_ps(__mmask8 __U, __m256 __A, __m256 __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_blend_ps(__U,__A,__W); 
}
__m128i test_mm_mask_blend_epi64(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_blend_epi64(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi64(__mmask8 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
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

__m128i test_mm_mask_cvtepi8_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.128
  return _mm_mask_cvtepi8_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.128
  return _mm_maskz_cvtepi8_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.256
  return _mm256_mask_cvtepi8_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.256
  return _mm256_maskz_cvtepi8_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepi8_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.128
  return _mm_mask_cvtepi8_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.128
  return _mm_maskz_cvtepi8_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.256
  return _mm256_mask_cvtepi8_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.256
  return _mm256_maskz_cvtepi8_epi64(__U, __A); 
}

__m128i test_mm_mask_cvtepi32_epi64(__m128i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.128
  return _mm_mask_cvtepi32_epi64(__W, __U, __X); 
}

__m128i test_mm_maskz_cvtepi32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.128
  return _mm_maskz_cvtepi32_epi64(__U, __X); 
}

__m256i test_mm256_mask_cvtepi32_epi64(__m256i __W, __mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.256
  return _mm256_mask_cvtepi32_epi64(__W, __U, __X); 
}

__m256i test_mm256_maskz_cvtepi32_epi64(__mmask8 __U, __m128i __X) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.256
  return _mm256_maskz_cvtepi32_epi64(__U, __X); 
}

__m128i test_mm_mask_cvtepi16_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.128
  return _mm_mask_cvtepi16_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.128
  return _mm_maskz_cvtepi16_epi32(__U, __A); 
}

__m256i test_mm256_mask_cvtepi16_epi32(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.256
  return _mm256_mask_cvtepi16_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi16_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.256
  return _mm256_maskz_cvtepi16_epi32(__U, __A); 
}

__m128i test_mm_mask_cvtepi16_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.128
  return _mm_mask_cvtepi16_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.128
  return _mm_maskz_cvtepi16_epi64(__U, __A); 
}

__m256i test_mm256_mask_cvtepi16_epi64(__m256i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.256
  return _mm256_mask_cvtepi16_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.256
  return _mm256_maskz_cvtepi16_epi64(__U, __A); 
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

__m128i test_mm_rol_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.128
  return _mm_rol_epi32(__A, 5); 
}

__m128i test_mm_mask_rol_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.128
  return _mm_mask_rol_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_rol_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.128
  return _mm_maskz_rol_epi32(__U, __A, 5); 
}

__m256i test_mm256_rol_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.256
  return _mm256_rol_epi32(__A, 5); 
}

__m256i test_mm256_mask_rol_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.256
  return _mm256_mask_rol_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_rol_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.256
  return _mm256_maskz_rol_epi32(__U, __A, 5); 
}

__m128i test_mm_rol_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.128
  return _mm_rol_epi64(__A, 5); 
}

__m128i test_mm_mask_rol_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.128
  return _mm_mask_rol_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_rol_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.128
  return _mm_maskz_rol_epi64(__U, __A, 5); 
}

__m256i test_mm256_rol_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.256
  return _mm256_rol_epi64(__A, 5); 
}

__m256i test_mm256_mask_rol_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.256
  return _mm256_mask_rol_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_rol_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.256
  return _mm256_maskz_rol_epi64(__U, __A, 5); 
}

__m128i test_mm_rolv_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.128
  return _mm_rolv_epi32(__A, __B); 
}

__m128i test_mm_mask_rolv_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.128
  return _mm_mask_rolv_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rolv_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.128
  return _mm_maskz_rolv_epi32(__U, __A, __B); 
}

__m256i test_mm256_rolv_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.256
  return _mm256_rolv_epi32(__A, __B); 
}

__m256i test_mm256_mask_rolv_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.256
  return _mm256_mask_rolv_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rolv_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.256
  return _mm256_maskz_rolv_epi32(__U, __A, __B); 
}

__m128i test_mm_rolv_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.128
  return _mm_rolv_epi64(__A, __B); 
}

__m128i test_mm_mask_rolv_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.128
  return _mm_mask_rolv_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rolv_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.128
  return _mm_maskz_rolv_epi64(__U, __A, __B); 
}

__m256i test_mm256_rolv_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.256
  return _mm256_rolv_epi64(__A, __B); 
}

__m256i test_mm256_mask_rolv_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.256
  return _mm256_mask_rolv_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rolv_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.256
  return _mm256_maskz_rolv_epi64(__U, __A, __B); 
}

__m128i test_mm_ror_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.128
  return _mm_ror_epi32(__A, 5); 
}

__m128i test_mm_mask_ror_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.128
  return _mm_mask_ror_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_ror_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.128
  return _mm_maskz_ror_epi32(__U, __A, 5); 
}

__m256i test_mm256_ror_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.256
  return _mm256_ror_epi32(__A, 5); 
}

__m256i test_mm256_mask_ror_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.256
  return _mm256_mask_ror_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_ror_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.256
  return _mm256_maskz_ror_epi32(__U, __A, 5); 
}

__m128i test_mm_ror_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.128
  return _mm_ror_epi64(__A, 5); 
}

__m128i test_mm_mask_ror_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.128
  return _mm_mask_ror_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_ror_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.128
  return _mm_maskz_ror_epi64(__U, __A, 5); 
}

__m256i test_mm256_ror_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.256
  return _mm256_ror_epi64(__A, 5); 
}

__m256i test_mm256_mask_ror_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.256
  return _mm256_mask_ror_epi64(__W, __U, __A,5); 
}

__m256i test_mm256_maskz_ror_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.256
  return _mm256_maskz_ror_epi64(__U, __A, 5); 
}


__m128i test_mm_rorv_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.128
  return _mm_rorv_epi32(__A, __B); 
}

__m128i test_mm_mask_rorv_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.128
  return _mm_mask_rorv_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rorv_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.128
  return _mm_maskz_rorv_epi32(__U, __A, __B); 
}

__m256i test_mm256_rorv_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.256
  return _mm256_rorv_epi32(__A, __B); 
}

__m256i test_mm256_mask_rorv_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.256
  return _mm256_mask_rorv_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rorv_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.256
  return _mm256_maskz_rorv_epi32(__U, __A, __B); 
}

__m128i test_mm_rorv_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.128
  return _mm_rorv_epi64(__A, __B); 
}

__m128i test_mm_mask_rorv_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.128
  return _mm_mask_rorv_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_rorv_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.128
  return _mm_maskz_rorv_epi64(__U, __A, __B); 
}

__m256i test_mm256_rorv_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.256
  return _mm256_rorv_epi64(__A, __B); 
}

__m256i test_mm256_mask_rorv_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.256
  return _mm256_mask_rorv_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_rorv_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.256
  return _mm256_maskz_rorv_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_sllv_epi64(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm_mask_sllv_epi64(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_sllv_epi64(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm_maskz_sllv_epi64(__U, __X, __Y); 
}

__m256i test_mm256_mask_sllv_epi64(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm256_mask_sllv_epi64(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_sllv_epi64(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm256_maskz_sllv_epi64(__U, __X, __Y); 
}

__m128i test_mm_mask_sllv_epi32(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm_mask_sllv_epi32(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_sllv_epi32(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm_maskz_sllv_epi32(__U, __X, __Y); 
}

__m256i test_mm256_mask_sllv_epi32(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm256_mask_sllv_epi32(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_sllv_epi32(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv
  return _mm256_maskz_sllv_epi32(__U, __X, __Y); 
}

__m128i test_mm_mask_srlv_epi64(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm_mask_srlv_epi64(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srlv_epi64(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm_maskz_srlv_epi64(__U, __X, __Y); 
}

__m256i test_mm256_mask_srlv_epi64(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm256_mask_srlv_epi64(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srlv_epi64(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm256_maskz_srlv_epi64(__U, __X, __Y); 
}

__m128i test_mm_mask_srlv_epi32(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm_mask_srlv_epi32(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srlv_epi32(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm_maskz_srlv_epi32(__U, __X, __Y); 
}

__m256i test_mm256_mask_srlv_epi32(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm256_mask_srlv_epi32(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srlv_epi32(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv
  return _mm256_maskz_srlv_epi32(__U, __X, __Y); 
}

__m128i test_mm_mask_srl_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d.128
  return _mm_mask_srl_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srl_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d.128
  return _mm_maskz_srl_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_srl_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d.256
  return _mm256_mask_srl_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srl_epi32(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d.256
  return _mm256_maskz_srl_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_srli_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.128
  return _mm_mask_srli_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_srli_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.128
  return _mm_maskz_srli_epi32(__U, __A, 5); 
}

__m256i test_mm256_mask_srli_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.256
  return _mm256_mask_srli_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_srli_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.256
  return _mm256_maskz_srli_epi32(__U, __A, 5); 
}

__m128i test_mm_mask_srl_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q.128
  return _mm_mask_srl_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srl_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q.128
  return _mm_maskz_srl_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_srl_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q.256
  return _mm256_mask_srl_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srl_epi64(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q.256
  return _mm256_maskz_srl_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_srli_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.128
  return _mm_mask_srli_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_srli_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.128
  return _mm_maskz_srli_epi64(__U, __A, 5); 
}

__m256i test_mm256_mask_srli_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.256
  return _mm256_mask_srli_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_srli_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.256
  return _mm256_maskz_srli_epi64(__U, __A, 5); 
}

__m128i test_mm_mask_srav_epi32(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav
  return _mm_mask_srav_epi32(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srav_epi32(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav
  return _mm_maskz_srav_epi32(__U, __X, __Y); 
}

__m256i test_mm256_mask_srav_epi32(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav
  return _mm256_mask_srav_epi32(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srav_epi32(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav
  return _mm256_maskz_srav_epi32(__U, __X, __Y); 
}

__m128i test_mm_srav_epi64(__m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q.128
  return _mm_srav_epi64(__X, __Y); 
}

__m128i test_mm_mask_srav_epi64(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q.128
  return _mm_mask_srav_epi64(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_srav_epi64(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q.128
  return _mm_maskz_srav_epi64(__U, __X, __Y); 
}

__m256i test_mm256_srav_epi64(__m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q.256
  return _mm256_srav_epi64(__X, __Y); 
}

__m256i test_mm256_mask_srav_epi64(__m256i __W, __mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q.256
  return _mm256_mask_srav_epi64(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_srav_epi64(__mmask8 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q.256
  return _mm256_maskz_srav_epi64(__U, __X, __Y); 
}

void test_mm_mask_store_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_store_epi32
  // CHECK: @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %{{.*}}, <4 x i32>* %{{.}}, i32 16, <4 x i1> %{{.*}})
  return _mm_mask_store_epi32(__P, __U, __A); 
}

void test_mm256_mask_store_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_store_epi32
  // CHECK: @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %{{.*}}, <8 x i32>* %{{.}}, i32 32, <8 x i1> %{{.*}})
  return _mm256_mask_store_epi32(__P, __U, __A); 
}

__m128i test_mm_mask_mov_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_mov_epi32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_mov_epi32(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_epi32
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_mov_epi32(__U, __A); 
}

__m256i test_mm256_mask_mov_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_epi32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_mov_epi32(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_epi32
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_mov_epi32(__U, __A); 
}

__m128i test_mm_mask_mov_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_mov_epi64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_mov_epi64(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_epi64
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_mov_epi64(__U, __A); 
}

__m256i test_mm256_mask_mov_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_epi64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_mov_epi64(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_epi64
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_mov_epi64(__U, __A); 
}

__m128i test_mm_mask_load_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_load_epi32(__W, __U, __P); 
}

__m128i test_mm_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_load_epi32(__U, __P); 
}

__m256i test_mm256_mask_load_epi32(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_load_epi32(__W, __U, __P); 
}

__m256i test_mm256_maskz_load_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_load_epi32(__U, __P); 
}

__m128i test_mm_mask_load_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_load_epi64(__W, __U, __P); 
}

__m128i test_mm_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_load_epi64(__U, __P); 
}

__m256i test_mm256_mask_load_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_load_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_load_epi64(__U, __P); 
}

void test_mm_mask_store_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_store_epi64
  // CHECK: @llvm.masked.store.v2i64.p0v2i64(<2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, i32 16, <2 x i1> %{{.*}})
  return _mm_mask_store_epi64(__P, __U, __A); 
}

void test_mm256_mask_store_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_store_epi64
  // CHECK: @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, i32 32, <4 x i1> %{{.*}})
  return _mm256_mask_store_epi64(__P, __U, __A); 
}

__m128d test_mm_mask_movedup_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_movedup_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_movedup_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_movedup_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_movedup_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_movedup_pd(__U, __A); 
}

__m256d test_mm256_mask_movedup_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_movedup_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_movedup_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_movedup_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_movedup_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_movedup_pd(__U, __A); 
}

__m128i test_mm_mask_set1_epi32(__m128i __O, __mmask8 __M) {
  // CHECK-LABEL: @test_mm_mask_set1_epi32
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.d.gpr.128
  return _mm_mask_set1_epi32(__O, __M, 5); 
}

__m128i test_mm_maskz_set1_epi32(__mmask8 __M) {
  // CHECK-LABEL: @test_mm_maskz_set1_epi32
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.d.gpr.128
  return _mm_maskz_set1_epi32(__M, 5); 
}

__m256i test_mm256_mask_set1_epi32(__m256i __O, __mmask8 __M) {
  // CHECK-LABEL: @test_mm256_mask_set1_epi32
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.d.gpr.256
  return _mm256_mask_set1_epi32(__O, __M, 5); 
}

__m256i test_mm256_maskz_set1_epi32(__mmask8 __M) {
  // CHECK-LABEL: @test_mm256_maskz_set1_epi32
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.d.gpr.256
  return _mm256_maskz_set1_epi32(__M, 5); 
}

__m128i test_mm_mask_set1_epi64(__m128i __O, __mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm_mask_set1_epi64
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.q.gpr.128
  return _mm_mask_set1_epi64(__O, __M, __A); 
}

__m128i test_mm_maskz_set1_epi64(__mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm_maskz_set1_epi64
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.q.gpr.128
  return _mm_maskz_set1_epi64(__M, __A); 
}

__m256i test_mm256_mask_set1_epi64(__m256i __O, __mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm256_mask_set1_epi64
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.q.gpr.256
  return _mm256_mask_set1_epi64(__O, __M, __A); 
}

__m256i test_mm256_maskz_set1_epi64(__mmask8 __M, long long __A) {
  // CHECK-LABEL: @test_mm256_maskz_set1_epi64
  // CHECK: @llvm.x86.avx512.mask.pbroadcast.q.gpr.256
  return _mm256_maskz_set1_epi64(__M, __A); 
}

__m128d test_mm_fixupimm_pd(__m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.128
  return _mm_fixupimm_pd(__A, __B, __C, 5); 
}

__m128d test_mm_mask_fixupimm_pd(__m128d __A, __mmask8 __U, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.128
  return _mm_mask_fixupimm_pd(__A, __U, __B, __C, 5); 
}

__m128d test_mm_maskz_fixupimm_pd(__mmask8 __U, __m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.128
  return _mm_maskz_fixupimm_pd(__U, __A, __B, __C, 5); 
}

__m256d test_mm256_fixupimm_pd(__m256d __A, __m256d __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.256
  return _mm256_fixupimm_pd(__A, __B, __C, 5); 
}

__m256d test_mm256_mask_fixupimm_pd(__m256d __A, __mmask8 __U, __m256d __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.256
  return _mm256_mask_fixupimm_pd(__A, __U, __B, __C, 5); 
}

__m256d test_mm256_maskz_fixupimm_pd(__mmask8 __U, __m256d __A, __m256d __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_fixupimm_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.256
  return _mm256_maskz_fixupimm_pd(__U, __A, __B, __C, 5); 
}

__m128 test_mm_fixupimm_ps(__m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.128
  return _mm_fixupimm_ps(__A, __B, __C, 5); 
}

__m128 test_mm_mask_fixupimm_ps(__m128 __A, __mmask8 __U, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.128
  return _mm_mask_fixupimm_ps(__A, __U, __B, __C, 5); 
}

__m128 test_mm_maskz_fixupimm_ps(__mmask8 __U, __m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.128
  return _mm_maskz_fixupimm_ps(__U, __A, __B, __C, 5); 
}

__m256 test_mm256_fixupimm_ps(__m256 __A, __m256 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.256
  return _mm256_fixupimm_ps(__A, __B, __C, 5); 
}

__m256 test_mm256_mask_fixupimm_ps(__m256 __A, __mmask8 __U, __m256 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.256
  return _mm256_mask_fixupimm_ps(__A, __U, __B, __C, 5); 
}

__m256 test_mm256_maskz_fixupimm_ps(__mmask8 __U, __m256 __A, __m256 __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_fixupimm_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.256
  return _mm256_maskz_fixupimm_ps(__U, __A, __B, __C, 5); 
}

__m128d test_mm_mask_load_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_load_pd(__W, __U, __P); 
}

__m128d test_mm_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 16, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_load_pd(__U, __P); 
}

__m256d test_mm256_mask_load_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_load_pd(__W, __U, __P); 
}

__m256d test_mm256_maskz_load_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 32, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_load_pd(__U, __P); 
}

__m128 test_mm_mask_load_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_load_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_load_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_load_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 16, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_load_ps(__U, __P); 
}

__m256 test_mm256_mask_load_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_load_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_load_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_load_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_load_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 32, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_load_ps(__U, __P); 
}

__m128i test_mm_mask_loadu_epi64(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_mask_loadu_epi64(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_epi64
  // CHECK: @llvm.masked.load.v2i64.p0v2i64(<2 x i64>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_maskz_loadu_epi64(__U, __P); 
}

__m256i test_mm256_mask_loadu_epi64(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_mask_loadu_epi64(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_epi64
  // CHECK: @llvm.masked.load.v4i64.p0v4i64(<4 x i64>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_maskz_loadu_epi64(__U, __P); 
}

__m128i test_mm_mask_loadu_epi32(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_mask_loadu_epi32(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_epi32
  // CHECK: @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_maskz_loadu_epi32(__U, __P); 
}

__m256i test_mm256_mask_loadu_epi32(__m256i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_mask_loadu_epi32(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi32(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_epi32
  // CHECK: @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_maskz_loadu_epi32(__U, __P); 
}

__m128d test_mm_mask_loadu_pd(__m128d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_mask_loadu_pd(__W, __U, __P); 
}

__m128d test_mm_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_pd
  // CHECK: @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %{{.*}}, i32 1, <2 x i1> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maskz_loadu_pd(__U, __P); 
}

__m256d test_mm256_mask_loadu_pd(__m256d __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_mask_loadu_pd(__W, __U, __P); 
}

__m256d test_mm256_maskz_loadu_pd(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_pd
  // CHECK: @llvm.masked.load.v4f64.p0v4f64(<4 x double>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maskz_loadu_pd(__U, __P); 
}

__m128 test_mm_mask_loadu_ps(__m128 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_mask_loadu_ps(__W, __U, __P); 
}

__m128 test_mm_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_ps
  // CHECK: @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %{{.*}}, i32 1, <4 x i1> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maskz_loadu_ps(__U, __P); 
}

__m256 test_mm256_mask_loadu_ps(__m256 __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_mask_loadu_ps(__W, __U, __P); 
}

__m256 test_mm256_maskz_loadu_ps(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_ps
  // CHECK: @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maskz_loadu_ps(__U, __P); 
}

void test_mm_mask_store_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_store_pd
  // CHECK: @llvm.masked.store.v2f64.p0v2f64(<2 x double> %{{.*}}, <2 x double>* %{{.*}}, i32 16, <2 x i1> %{{.*}})
  return _mm_mask_store_pd(__P, __U, __A); 
}

void test_mm256_mask_store_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_store_pd
  // CHECK: @llvm.masked.store.v4f64.p0v4f64(<4 x double> %{{.*}}, <4 x double>* %{{.*}}, i32 32, <4 x i1> %{{.*}})
  return _mm256_mask_store_pd(__P, __U, __A); 
}

void test_mm_mask_store_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_store_ps
  // CHECK: @llvm.masked.store.v4f32.p0v4f32(<4 x float> %{{.*}}, <4 x float>* %{{.*}}, i32 16, <4 x i1> %{{.*}})
  return _mm_mask_store_ps(__P, __U, __A); 
}

void test_mm256_mask_store_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_store_ps
  // CHECK: @llvm.masked.store.v8f32.p0v8f32(<8 x float> %{{.*}}, <8 x float>* %{{.*}}, i32 32, <8 x i1> %{{.*}})
  return _mm256_mask_store_ps(__P, __U, __A); 
}

void test_mm_mask_storeu_epi64(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_epi64
  // CHECK: @llvm.masked.store.v2i64.p0v2i64(<2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, i32 1, <2 x i1> %{{.*}})
  return _mm_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi64(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_epi64
  // CHECK: @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_epi64(__P, __U, __A); 
}

void test_mm_mask_storeu_epi32(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_epi32
  // CHECK: @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %{{.*}}, <4 x i32>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi32(void *__P, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_epi32
  // CHECK: @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %{{.*}}, <8 x i32>* %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_epi32(__P, __U, __A); 
}

void test_mm_mask_storeu_pd(void *__P, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_pd
  // CHECK: @llvm.masked.store.v2f64.p0v2f64(<2 x double> %{{.*}}, <2 x double>* %{{.*}}, i32 1, <2 x i1> %{{.*}})
  return _mm_mask_storeu_pd(__P, __U, __A); 
}

void test_mm256_mask_storeu_pd(void *__P, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_pd
  // CHECK: @llvm.masked.store.v4f64.p0v4f64(<4 x double> %{{.*}}, <4 x double>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm256_mask_storeu_pd(__P, __U, __A); 
}

void test_mm_mask_storeu_ps(void *__P, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_ps
  // CHECK: @llvm.masked.store.v4f32.p0v4f32(<4 x float> %{{.*}}, <4 x float>* %{{.*}}, i32 1, <4 x i1> %{{.*}})
  return _mm_mask_storeu_ps(__P, __U, __A); 
}

void test_mm256_mask_storeu_ps(void *__P, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_ps
  // CHECK: @llvm.masked.store.v8f32.p0v8f32(<8 x float> %{{.*}}, <8 x float>* %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm256_mask_storeu_ps(__P, __U, __A); 
}

__m128d test_mm_mask_unpackhi_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_unpackhi_pd(__W, __U, __A, __B); 
}

__m128d test_mm_maskz_unpackhi_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_unpackhi_pd(__U, __A, __B); 
}

__m256d test_mm256_mask_unpackhi_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_pd
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_unpackhi_pd(__W, __U, __A, __B); 
}

__m256d test_mm256_maskz_unpackhi_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_pd
  // CHECK: shufflevector <8 x i1> %{{.*}}, <8 x i1> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_unpackhi_pd(__U, __A, __B); 
}

__m128 test_mm_mask_unpackhi_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}} <4 x float> %{{.*}}
  return _mm_mask_unpackhi_ps(__W, __U, __A, __B); 
}

__m128 test_mm_maskz_unpackhi_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}} <4 x float> %{{.*}}
  return _mm_maskz_unpackhi_ps(__U, __A, __B); 
}

__m256 test_mm256_mask_unpackhi_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_unpackhi_ps(__W, __U, __A, __B); 
}

__m256 test_mm256_maskz_unpackhi_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_unpackhi_ps(__U, __A, __B); 
}

__m128d test_mm_mask_unpacklo_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_unpacklo_pd(__W, __U, __A, __B); 
}

__m128d test_mm_maskz_unpacklo_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_unpacklo_pd(__U, __A, __B); 
}

__m256d test_mm256_mask_unpacklo_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_unpacklo_pd(__W, __U, __A, __B); 
}

__m256d test_mm256_maskz_unpacklo_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}} <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_unpacklo_pd(__U, __A, __B); 
}

__m128 test_mm_mask_unpacklo_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_unpacklo_ps(__W, __U, __A, __B); 
}

__m128 test_mm_maskz_unpacklo_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_unpacklo_ps(__U, __A, __B); 
}

__m256 test_mm256_mask_unpacklo_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_unpacklo_ps(__W, __U, __A, __B); 
}

__m256 test_mm256_maskz_unpacklo_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_unpacklo_ps(__U, __A, __B); 
}

__m128d test_mm_rcp14_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.128
  return _mm_rcp14_pd(__A); 
}

__m128d test_mm_mask_rcp14_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.128
  return _mm_mask_rcp14_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_rcp14_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.128
  return _mm_maskz_rcp14_pd(__U, __A); 
}

__m256d test_mm256_rcp14_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.256
  return _mm256_rcp14_pd(__A); 
}

__m256d test_mm256_mask_rcp14_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.256
  return _mm256_mask_rcp14_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_rcp14_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_rcp14_pd
  // CHECK: @llvm.x86.avx512.rcp14.pd.256
  return _mm256_maskz_rcp14_pd(__U, __A); 
}

__m128 test_mm_rcp14_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.128
  return _mm_rcp14_ps(__A); 
}

__m128 test_mm_mask_rcp14_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.128
  return _mm_mask_rcp14_ps(__W, __U, __A); 
}

__m128 test_mm_maskz_rcp14_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.128
  return _mm_maskz_rcp14_ps(__U, __A); 
}

__m256 test_mm256_rcp14_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.256
  return _mm256_rcp14_ps(__A); 
}

__m256 test_mm256_mask_rcp14_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.256
  return _mm256_mask_rcp14_ps(__W, __U, __A); 
}

__m256 test_mm256_maskz_rcp14_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_rcp14_ps
  // CHECK: @llvm.x86.avx512.rcp14.ps.256
  return _mm256_maskz_rcp14_ps(__U, __A); 
}

__m128d test_mm_mask_permute_pd(__m128d __W, __mmask8 __U, __m128d __X) {
  // CHECK-LABEL: @test_mm_mask_permute_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_permute_pd(__W, __U, __X, 1); 
}

__m128d test_mm_maskz_permute_pd(__mmask8 __U, __m128d __X) {
  // CHECK-LABEL: @test_mm_maskz_permute_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_permute_pd(__U, __X, 1); 
}

__m256d test_mm256_mask_permute_pd(__m256d __W, __mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_mask_permute_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permute_pd(__W, __U, __X, 5); 
}

__m256d test_mm256_maskz_permute_pd(__mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_maskz_permute_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permute_pd(__U, __X, 5); 
}

__m128 test_mm_mask_permute_ps(__m128 __W, __mmask8 __U, __m128 __X) {
  // CHECK-LABEL: @test_mm_mask_permute_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_permute_ps(__W, __U, __X, 0x1b); 
}

__m128 test_mm_maskz_permute_ps(__mmask8 __U, __m128 __X) {
  // CHECK-LABEL: @test_mm_maskz_permute_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_permute_ps(__U, __X, 0x1b); 
}

__m256 test_mm256_mask_permute_ps(__m256 __W, __mmask8 __U, __m256 __X) {
  // CHECK-LABEL: @test_mm256_mask_permute_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_permute_ps(__W, __U, __X, 0x1b); 
}

__m256 test_mm256_maskz_permute_ps(__mmask8 __U, __m256 __X) {
  // CHECK-LABEL: @test_mm256_maskz_permute_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_permute_ps(__U, __X, 0x1b); 
}

__m128d test_mm_mask_permutevar_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd
  return _mm_mask_permutevar_pd(__W, __U, __A, __C); 
}

__m128d test_mm_maskz_permutevar_pd(__mmask8 __U, __m128d __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd
  return _mm_maskz_permutevar_pd(__U, __A, __C); 
}

__m256d test_mm256_mask_permutevar_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd.256
  return _mm256_mask_permutevar_pd(__W, __U, __A, __C); 
}

__m256d test_mm256_maskz_permutevar_pd(__mmask8 __U, __m256d __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd.256
  return _mm256_maskz_permutevar_pd(__U, __A, __C); 
}

__m128 test_mm_mask_permutevar_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps
  return _mm_mask_permutevar_ps(__W, __U, __A, __C); 
}

__m128 test_mm_maskz_permutevar_ps(__mmask8 __U, __m128 __A, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps
  return _mm_maskz_permutevar_ps(__U, __A, __C); 
}

__m256 test_mm256_mask_permutevar_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps.256
  return _mm256_mask_permutevar_ps(__W, __U, __A, __C); 
}

__m256 test_mm256_maskz_permutevar_ps(__mmask8 __U, __m256 __A, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps.256
  return _mm256_maskz_permutevar_ps(__U, __A, __C); 
}

__mmask8 test_mm_test_epi32_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_test_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestm.d.128
  return _mm_test_epi32_mask(__A, __B); 
}

__mmask8 test_mm_mask_test_epi32_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_test_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestm.d.128
  return _mm_mask_test_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm256_test_epi32_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_test_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestm.d.256
  return _mm256_test_epi32_mask(__A, __B); 
}

__mmask8 test_mm256_mask_test_epi32_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_test_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestm.d.256
  return _mm256_mask_test_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm_test_epi64_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_test_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestm.q.128
  return _mm_test_epi64_mask(__A, __B); 
}

__mmask8 test_mm_mask_test_epi64_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_test_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestm.q.128
  return _mm_mask_test_epi64_mask(__U, __A, __B); 
}

__mmask8 test_mm256_test_epi64_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_test_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestm.q.256
  return _mm256_test_epi64_mask(__A, __B); 
}

__mmask8 test_mm256_mask_test_epi64_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_test_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestm.q.256
  return _mm256_mask_test_epi64_mask(__U, __A, __B); 
}

__mmask8 test_mm_testn_epi32_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_testn_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestnm.d.128
  return _mm_testn_epi32_mask(__A, __B); 
}

__mmask8 test_mm_mask_testn_epi32_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_testn_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestnm.d.128
  return _mm_mask_testn_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm256_testn_epi32_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_testn_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestnm.d.256
  return _mm256_testn_epi32_mask(__A, __B); 
}

__mmask8 test_mm256_mask_testn_epi32_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_testn_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestnm.d.256
  return _mm256_mask_testn_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm_testn_epi64_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_testn_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestnm.q.128
  return _mm_testn_epi64_mask(__A, __B); 
}

__mmask8 test_mm_mask_testn_epi64_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_testn_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestnm.q.128
  return _mm_mask_testn_epi64_mask(__U, __A, __B); 
}

__mmask8 test_mm256_testn_epi64_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_testn_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestnm.q.256
  return _mm256_testn_epi64_mask(__A, __B); 
}

__mmask8 test_mm256_mask_testn_epi64_mask(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_testn_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestnm.q.256
  return _mm256_mask_testn_epi64_mask(__U, __A, __B); 
}
__m128i test_mm_mask_unpackhi_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_unpackhi_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_unpackhi_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_unpackhi_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_unpackhi_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_unpackhi_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_unpackhi_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_unpackhi_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_unpackhi_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_unpacklo_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_unpacklo_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_unpacklo_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_unpacklo_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_unpacklo_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 2>
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_unpacklo_epi64(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_unpacklo_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_unpacklo_epi64(__U, __A, __B); 
}

__m128i test_mm_mask_sra_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d.128
  return _mm_mask_sra_epi32(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sra_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d.128
  return _mm_maskz_sra_epi32(__U, __A, __B); 
}

__m256i test_mm256_mask_sra_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d.256
  return _mm256_mask_sra_epi32(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sra_epi32(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d.256
  return _mm256_maskz_sra_epi32(__U, __A, __B); 
}

__m128i test_mm_mask_srai_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.128
  return _mm_mask_srai_epi32(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_srai_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.128
  return _mm_maskz_srai_epi32(__U, __A, 5); 
}

__m256i test_mm256_mask_srai_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.256
  return _mm256_mask_srai_epi32(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_srai_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.256
  return _mm256_maskz_srai_epi32(__U, __A, 5); 
}

__m128i test_mm_sra_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q.128
  return _mm_sra_epi64(__A, __B); 
}

__m128i test_mm_mask_sra_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q.128
  return _mm_mask_sra_epi64(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sra_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q.128
  return _mm_maskz_sra_epi64(__U, __A, __B); 
}

__m256i test_mm256_sra_epi64(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q.256
  return _mm256_sra_epi64(__A, __B); 
}

__m256i test_mm256_mask_sra_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q.256
  return _mm256_mask_sra_epi64(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sra_epi64(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q.256
  return _mm256_maskz_sra_epi64(__U, __A, __B); 
}

__m128i test_mm_srai_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.128
  return _mm_srai_epi64(__A, 5); 
}

__m128i test_mm_mask_srai_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.128
  return _mm_mask_srai_epi64(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_srai_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.128
  return _mm_maskz_srai_epi64(__U, __A, 5); 
}

__m256i test_mm256_srai_epi64(__m256i __A) {
  // CHECK-LABEL: @test_mm256_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.256
  return _mm256_srai_epi64(__A, 5); 
}

__m256i test_mm256_mask_srai_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.256
  return _mm256_mask_srai_epi64(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_srai_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.256
  return _mm256_maskz_srai_epi64(__U, __A, 5); 
}

__m128i test_mm_ternarylogic_epi32(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.mask.pternlog.d.128
  return _mm_ternarylogic_epi32(__A, __B, __C, 4); 
}

__m128i test_mm_mask_ternarylogic_epi32(__m128i __A, __mmask8 __U, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.mask.pternlog.d.128
  return _mm_mask_ternarylogic_epi32(__A, __U, __B, __C, 4); 
}

__m128i test_mm_maskz_ternarylogic_epi32(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.maskz.pternlog.d.128
  return _mm_maskz_ternarylogic_epi32(__U, __A, __B, __C, 4); 
}

__m256i test_mm256_ternarylogic_epi32(__m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.mask.pternlog.d.256
  return _mm256_ternarylogic_epi32(__A, __B, __C, 4); 
}

__m256i test_mm256_mask_ternarylogic_epi32(__m256i __A, __mmask8 __U, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.mask.pternlog.d.256
  return _mm256_mask_ternarylogic_epi32(__A, __U, __B, __C, 4); 
}

__m256i test_mm256_maskz_ternarylogic_epi32(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.maskz.pternlog.d.256
  return _mm256_maskz_ternarylogic_epi32(__U, __A, __B, __C, 4); 
}

__m128i test_mm_ternarylogic_epi64(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.mask.pternlog.q.128
  return _mm_ternarylogic_epi64(__A, __B, __C, 4); 
}

__m128i test_mm_mask_ternarylogic_epi64(__m128i __A, __mmask8 __U, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.mask.pternlog.q.128
  return _mm_mask_ternarylogic_epi64(__A, __U, __B, __C, 4); 
}

__m128i test_mm_maskz_ternarylogic_epi64(__mmask8 __U, __m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.maskz.pternlog.q.128
  return _mm_maskz_ternarylogic_epi64(__U, __A, __B, __C, 4); 
}

__m256i test_mm256_ternarylogic_epi64(__m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.mask.pternlog.q.256
  return _mm256_ternarylogic_epi64(__A, __B, __C, 4); 
}

__m256i test_mm256_mask_ternarylogic_epi64(__m256i __A, __mmask8 __U, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_mask_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.mask.pternlog.q.256
  return _mm256_mask_ternarylogic_epi64(__A, __U, __B, __C, 4); 
}

__m256i test_mm256_maskz_ternarylogic_epi64(__mmask8 __U, __m256i __A, __m256i __B, __m256i __C) {
  // CHECK-LABEL: @test_mm256_maskz_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.maskz.pternlog.q.256
  return _mm256_maskz_ternarylogic_epi64(__U, __A, __B, __C, 4); 
}
__m256 test_mm256_shuffle_f32x4(__m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_shuffle_f32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.f32x4
  return _mm256_shuffle_f32x4(__A, __B, 3); 
}

__m256 test_mm256_mask_shuffle_f32x4(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_f32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.f32x4
  return _mm256_mask_shuffle_f32x4(__W, __U, __A, __B, 3); 
}

__m256 test_mm256_maskz_shuffle_f32x4(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_f32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.f32x4
  return _mm256_maskz_shuffle_f32x4(__U, __A, __B, 3); 
}

__m256d test_mm256_shuffle_f64x2(__m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_shuffle_f64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.f64x2
  return _mm256_shuffle_f64x2(__A, __B, 3); 
}

__m256d test_mm256_mask_shuffle_f64x2(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_f64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.f64x2
  return _mm256_mask_shuffle_f64x2(__W, __U, __A, __B, 3); 
}

__m256d test_mm256_maskz_shuffle_f64x2(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_f64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.f64x2
  return _mm256_maskz_shuffle_f64x2(__U, __A, __B, 3); 
}

__m256i test_mm256_shuffle_i32x4(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_shuffle_i32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.i32x4
  return _mm256_shuffle_i32x4(__A, __B, 3); 
}

__m256i test_mm256_mask_shuffle_i32x4(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_i32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.i32x4
  return _mm256_mask_shuffle_i32x4(__W, __U, __A, __B, 3); 
}

__m256i test_mm256_maskz_shuffle_i32x4(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_i32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.i32x4
  return _mm256_maskz_shuffle_i32x4(__U, __A, __B, 3); 
}

__m256i test_mm256_shuffle_i64x2(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_shuffle_i64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.i64x2
  return _mm256_shuffle_i64x2(__A, __B, 3); 
}

__m256i test_mm256_mask_shuffle_i64x2(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_i64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.i64x2
  return _mm256_mask_shuffle_i64x2(__W, __U, __A, __B, 3); 
}

__m256i test_mm256_maskz_shuffle_i64x2(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_i64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.i64x2
  return _mm256_maskz_shuffle_i64x2(__U, __A, __B, 3); 
}

__m128d test_mm_mask_shuffle_pd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_shuffle_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_shuffle_pd(__W, __U, __A, __B, 3); 
}

__m128d test_mm_maskz_shuffle_pd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 3>
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_shuffle_pd(__U, __A, __B, 3); 
}

__m256d test_mm256_mask_shuffle_pd(__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_shuffle_pd(__W, __U, __A, __B, 3); 
}

__m256d test_mm256_maskz_shuffle_pd(__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 2, i32 6>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_shuffle_pd(__U, __A, __B, 3); 
}

__m128 test_mm_mask_shuffle_ps(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_shuffle_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_shuffle_ps(__W, __U, __A, __B, 4); 
}

__m128 test_mm_maskz_shuffle_ps(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 4, i32 4>
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_shuffle_ps(__U, __A, __B, 4); 
}

__m256 test_mm256_mask_shuffle_ps(__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 8, i32 4, i32 5, i32 12, i32 12>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_shuffle_ps(__W, __U, __A, __B, 4); 
}

__m256 test_mm256_maskz_shuffle_ps(__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 8, i32 8, i32 4, i32 5, i32 12, i32 12>
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_shuffle_ps(__U, __A, __B, 4); 
}

__m128d test_mm_rsqrt14_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.128
  return _mm_rsqrt14_pd(__A); 
}

__m128d test_mm_mask_rsqrt14_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.128
  return _mm_mask_rsqrt14_pd(__W, __U, __A); 
}

__m128d test_mm_maskz_rsqrt14_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.128
  return _mm_maskz_rsqrt14_pd(__U, __A); 
}

__m256d test_mm256_rsqrt14_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.256
  return _mm256_rsqrt14_pd(__A); 
}

__m256d test_mm256_mask_rsqrt14_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.256
  return _mm256_mask_rsqrt14_pd(__W, __U, __A); 
}

__m256d test_mm256_maskz_rsqrt14_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.256
  return _mm256_maskz_rsqrt14_pd(__U, __A); 
}

__m128 test_mm_rsqrt14_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.128
  return _mm_rsqrt14_ps(__A); 
}

__m128 test_mm_mask_rsqrt14_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.128
  return _mm_mask_rsqrt14_ps(__W, __U, __A); 
}

__m128 test_mm_maskz_rsqrt14_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.128
  return _mm_maskz_rsqrt14_ps(__U, __A); 
}

__m256 test_mm256_rsqrt14_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.256
  return _mm256_rsqrt14_ps(__A); 
}

__m256 test_mm256_mask_rsqrt14_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.256
  return _mm256_mask_rsqrt14_ps(__W, __U, __A); 
}

__m256 test_mm256_maskz_rsqrt14_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.256
  return _mm256_maskz_rsqrt14_ps(__U, __A); 
}

__m256 test_mm256_broadcast_f32x4(__m128 __A) {
  // CHECK-LABEL: @test_mm256_broadcast_f32x4
  // CHECK: @llvm.x86.avx512.mask.broadcastf32x4
  return _mm256_broadcast_f32x4(__A); 
}

__m256 test_mm256_mask_broadcast_f32x4(__m256 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_f32x4
  // CHECK: @llvm.x86.avx512.mask.broadcastf32x4
  return _mm256_mask_broadcast_f32x4(__O, __M, __A); 
}

__m256 test_mm256_maskz_broadcast_f32x4(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_f32x4
  // CHECK: @llvm.x86.avx512.mask.broadcastf32x4
  return _mm256_maskz_broadcast_f32x4(__M, __A); 
}

__m256i test_mm256_broadcast_i32x4(__m128i __A) {
  // CHECK-LABEL: @test_mm256_broadcast_i32x4
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x4
  return _mm256_broadcast_i32x4(__A); 
}

__m256i test_mm256_mask_broadcast_i32x4(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcast_i32x4
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x4
  return _mm256_mask_broadcast_i32x4(__O, __M, __A); 
}

__m256i test_mm256_maskz_broadcast_i32x4(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcast_i32x4
  // CHECK: @llvm.x86.avx512.mask.broadcasti32x4
  return _mm256_maskz_broadcast_i32x4(__M, __A); 
}

__m256d test_mm256_mask_broadcastsd_pd(__m256d __O, __mmask8 __M, __m128d __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_broadcastsd_pd(__O, __M, __A);
}

__m256d test_mm256_maskz_broadcastsd_pd(__mmask8 __M, __m128d __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastsd_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_broadcastsd_pd(__M, __A);
}

__m128 test_mm_mask_broadcastss_ps(__m128 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_broadcastss_ps(__O, __M, __A);
}

__m128 test_mm_maskz_broadcastss_ps(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_broadcastss_ps(__M, __A);
}

__m256 test_mm256_mask_broadcastss_ps(__m256 __O, __mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_broadcastss_ps(__O, __M, __A);
}

__m256 test_mm256_maskz_broadcastss_ps(__mmask8 __M, __m128 __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastss_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_broadcastss_ps(__M, __A);
}

__m128i test_mm_mask_broadcastd_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_broadcastd_epi32(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastd_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_broadcastd_epi32(__M, __A);
}

__m256i test_mm256_mask_broadcastd_epi32(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_broadcastd_epi32(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastd_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastd_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_broadcastd_epi32(__M, __A);
}

__m128i test_mm_mask_broadcastq_epi64(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_broadcastq_epi64(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastq_epi64(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_broadcastq_epi64(__M, __A);
}

__m256i test_mm256_mask_broadcastq_epi64(__m256i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_broadcastq_epi64(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastq_epi64(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastq_epi64
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_broadcastq_epi64(__M, __A);
}

__m128i test_mm_cvtsepi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.128
  return _mm_cvtsepi32_epi8(__A); 
}

__m128i test_mm_mask_cvtsepi32_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.128
  return _mm_mask_cvtsepi32_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi32_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.128
  return _mm_maskz_cvtsepi32_epi8(__M, __A); 
}

void test_mm_mask_cvtsepi32_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.mem.128
  return _mm_mask_cvtsepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.256
  return _mm256_cvtsepi32_epi8(__A); 
}

__m128i test_mm256_mask_cvtsepi32_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.256
  return _mm256_mask_cvtsepi32_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi32_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.256
  return _mm256_maskz_cvtsepi32_epi8(__M, __A); 
}

void test_mm256_mask_cvtsepi32_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.mem.256
  return _mm256_mask_cvtsepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtsepi32_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.128
  return _mm_cvtsepi32_epi16(__A); 
}

__m128i test_mm_mask_cvtsepi32_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.128
  return _mm_mask_cvtsepi32_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi32_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.128
  return _mm_maskz_cvtsepi32_epi16(__M, __A); 
}

void test_mm_mask_cvtsepi32_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.mem.128
  return _mm_mask_cvtsepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi32_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.256
  return _mm256_cvtsepi32_epi16(__A); 
}

__m128i test_mm256_mask_cvtsepi32_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.256
  return _mm256_mask_cvtsepi32_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi32_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.256
  return _mm256_maskz_cvtsepi32_epi16(__M, __A); 
}

void test_mm256_mask_cvtsepi32_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.mem.256
  return _mm256_mask_cvtsepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtsepi64_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.128
  return _mm_cvtsepi64_epi8(__A); 
}

__m128i test_mm_mask_cvtsepi64_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.128
  return _mm_mask_cvtsepi64_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi64_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.128
  return _mm_maskz_cvtsepi64_epi8(__M, __A); 
}

void test_mm_mask_cvtsepi64_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.mem.128
  return _mm_mask_cvtsepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi64_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.256
  return _mm256_cvtsepi64_epi8(__A); 
}

__m128i test_mm256_mask_cvtsepi64_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.256
  return _mm256_mask_cvtsepi64_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi64_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.256
  return _mm256_maskz_cvtsepi64_epi8(__M, __A); 
}

void test_mm256_mask_cvtsepi64_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.mem.256
  return _mm256_mask_cvtsepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtsepi64_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.128
  return _mm_cvtsepi64_epi32(__A); 
}

__m128i test_mm_mask_cvtsepi64_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.128
  return _mm_mask_cvtsepi64_epi32(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi64_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.128
  return _mm_maskz_cvtsepi64_epi32(__M, __A); 
}

void test_mm_mask_cvtsepi64_storeu_epi32(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.mem.128
  return _mm_mask_cvtsepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi64_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.256
  return _mm256_cvtsepi64_epi32(__A); 
}

__m128i test_mm256_mask_cvtsepi64_epi32(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.256
  return _mm256_mask_cvtsepi64_epi32(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi64_epi32(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.256
  return _mm256_maskz_cvtsepi64_epi32(__M, __A); 
}

void test_mm256_mask_cvtsepi64_storeu_epi32(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.mem.256
  return _mm256_mask_cvtsepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm_cvtsepi64_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.128
  return _mm_cvtsepi64_epi16(__A); 
}

__m128i test_mm_mask_cvtsepi64_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.128
  return _mm_mask_cvtsepi64_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi64_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.128
  return _mm_maskz_cvtsepi64_epi16(__M, __A); 
}

void test_mm_mask_cvtsepi64_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtsepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.mem.128
  return _mm_mask_cvtsepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtsepi64_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.256
  return _mm256_cvtsepi64_epi16(__A); 
}

__m128i test_mm256_mask_cvtsepi64_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.256
  return _mm256_mask_cvtsepi64_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi64_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.256
  return _mm256_maskz_cvtsepi64_epi16(__M, __A); 
}

void test_mm256_mask_cvtsepi64_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtsepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.mem.256
  return _mm256_mask_cvtsepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtusepi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.128
  return _mm_cvtusepi32_epi8(__A); 
}

__m128i test_mm_mask_cvtusepi32_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.128
  return _mm_mask_cvtusepi32_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi32_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.128
  return _mm_maskz_cvtusepi32_epi8(__M, __A); 
}

void test_mm_mask_cvtusepi32_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.mem.128
  return _mm_mask_cvtusepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.256
  return _mm256_cvtusepi32_epi8(__A); 
}

__m128i test_mm256_mask_cvtusepi32_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.256
  return _mm256_mask_cvtusepi32_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi32_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.256
  return _mm256_maskz_cvtusepi32_epi8(__M, __A); 
}

void test_mm256_mask_cvtusepi32_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.mem.256
  return _mm256_mask_cvtusepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtusepi32_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.128
  return _mm_cvtusepi32_epi16(__A); 
}

__m128i test_mm_mask_cvtusepi32_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.128
  return _mm_mask_cvtusepi32_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi32_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.128
  return _mm_maskz_cvtusepi32_epi16(__M, __A); 
}

void test_mm_mask_cvtusepi32_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.mem.128
  return _mm_mask_cvtusepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi32_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.256
  return _mm256_cvtusepi32_epi16(__A); 
}

__m128i test_mm256_mask_cvtusepi32_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.256
  return _mm256_mask_cvtusepi32_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi32_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.256
  return _mm256_maskz_cvtusepi32_epi16(__M, __A); 
}

void test_mm256_mask_cvtusepi32_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.mem.256
  return _mm256_mask_cvtusepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtusepi64_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.128
  return _mm_cvtusepi64_epi8(__A); 
}

__m128i test_mm_mask_cvtusepi64_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.128
  return _mm_mask_cvtusepi64_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi64_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.128
  return _mm_maskz_cvtusepi64_epi8(__M, __A); 
}

void test_mm_mask_cvtusepi64_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.mem.128
  return _mm_mask_cvtusepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi64_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.256
  return _mm256_cvtusepi64_epi8(__A); 
}

__m128i test_mm256_mask_cvtusepi64_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.256
  return _mm256_mask_cvtusepi64_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi64_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.256
  return _mm256_maskz_cvtusepi64_epi8(__M, __A); 
}

void test_mm256_mask_cvtusepi64_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.mem.256
  return _mm256_mask_cvtusepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtusepi64_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.128
  return _mm_cvtusepi64_epi32(__A); 
}

__m128i test_mm_mask_cvtusepi64_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.128
  return _mm_mask_cvtusepi64_epi32(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi64_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.128
  return _mm_maskz_cvtusepi64_epi32(__M, __A); 
}

void test_mm_mask_cvtusepi64_storeu_epi32(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.mem.128
  return _mm_mask_cvtusepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi64_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.256
  return _mm256_cvtusepi64_epi32(__A); 
}

__m128i test_mm256_mask_cvtusepi64_epi32(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.256
  return _mm256_mask_cvtusepi64_epi32(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi64_epi32(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.256
  return _mm256_maskz_cvtusepi64_epi32(__M, __A); 
}

void test_mm256_mask_cvtusepi64_storeu_epi32(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.mem.256
  return _mm256_mask_cvtusepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm_cvtusepi64_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.128
  return _mm_cvtusepi64_epi16(__A); 
}

__m128i test_mm_mask_cvtusepi64_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.128
  return _mm_mask_cvtusepi64_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi64_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.128
  return _mm_maskz_cvtusepi64_epi16(__M, __A); 
}

void test_mm_mask_cvtusepi64_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtusepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.mem.128
  return _mm_mask_cvtusepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtusepi64_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.256
  return _mm256_cvtusepi64_epi16(__A); 
}

__m128i test_mm256_mask_cvtusepi64_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.256
  return _mm256_mask_cvtusepi64_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi64_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.256
  return _mm256_maskz_cvtusepi64_epi16(__M, __A); 
}

void test_mm256_mask_cvtusepi64_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtusepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.mem.256
  return _mm256_mask_cvtusepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtepi32_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.128
  return _mm_cvtepi32_epi8(__A); 
}

__m128i test_mm_mask_cvtepi32_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.128
  return _mm_mask_cvtepi32_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi32_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.128
  return _mm_maskz_cvtepi32_epi8(__M, __A); 
}

void test_mm_mask_cvtepi32_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.mem.128
  return _mm_mask_cvtepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtepi32_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.256
  return _mm256_cvtepi32_epi8(__A); 
}

__m128i test_mm256_mask_cvtepi32_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.256
  return _mm256_mask_cvtepi32_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi32_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.256
  return _mm256_maskz_cvtepi32_epi8(__M, __A); 
}

void test_mm256_mask_cvtepi32_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.mem.256
  return _mm256_mask_cvtepi32_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtepi32_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.128
  return _mm_cvtepi32_epi16(__A); 
}

__m128i test_mm_mask_cvtepi32_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.128
  return _mm_mask_cvtepi32_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi32_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.128
  return _mm_maskz_cvtepi32_epi16(__M, __A); 
}

void test_mm_mask_cvtepi32_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.mem.128
  return _mm_mask_cvtepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtepi32_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.256
  return _mm256_cvtepi32_epi16(__A); 
}

__m128i test_mm256_mask_cvtepi32_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.256
  return _mm256_mask_cvtepi32_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi32_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.256
  return _mm256_maskz_cvtepi32_epi16(__M, __A); 
}

void test_mm256_mask_cvtepi32_storeu_epi16(void *  __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.mem.256
  return _mm256_mask_cvtepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm_cvtepi64_epi8(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.128
  return _mm_cvtepi64_epi8(__A); 
}

__m128i test_mm_mask_cvtepi64_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.128
  return _mm_mask_cvtepi64_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi64_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.128
  return _mm_maskz_cvtepi64_epi8(__M, __A); 
}

void test_mm_mask_cvtepi64_storeu_epi8(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.mem.128
  return _mm_mask_cvtepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm256_cvtepi64_epi8(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.256
  return _mm256_cvtepi64_epi8(__A); 
}

__m128i test_mm256_mask_cvtepi64_epi8(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.256
  return _mm256_mask_cvtepi64_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi64_epi8(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.256
  return _mm256_maskz_cvtepi64_epi8(__M, __A); 
}

void test_mm256_mask_cvtepi64_storeu_epi8(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.mem.256
  return _mm256_mask_cvtepi64_storeu_epi8(__P, __M, __A); 
}

__m128i test_mm_cvtepi64_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.128
  return _mm_cvtepi64_epi32(__A); 
}

__m128i test_mm_mask_cvtepi64_epi32(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.128
  return _mm_mask_cvtepi64_epi32(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi64_epi32(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.128
  return _mm_maskz_cvtepi64_epi32(__M, __A); 
}

void test_mm_mask_cvtepi64_storeu_epi32(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.mem.128
  return _mm_mask_cvtepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm256_cvtepi64_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.256
  return _mm256_cvtepi64_epi32(__A); 
}

__m128i test_mm256_mask_cvtepi64_epi32(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.256
  return _mm256_mask_cvtepi64_epi32(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi64_epi32(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.256
  return _mm256_maskz_cvtepi64_epi32(__M, __A); 
}

void test_mm256_mask_cvtepi64_storeu_epi32(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.mem.256
  return _mm256_mask_cvtepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm_cvtepi64_epi16(__m128i __A) {
  // CHECK-LABEL: @test_mm_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.128
  return _mm_cvtepi64_epi16(__A); 
}

__m128i test_mm_mask_cvtepi64_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.128
  return _mm_mask_cvtepi64_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi64_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.128
  return _mm_maskz_cvtepi64_epi16(__M, __A); 
}

void test_mm_mask_cvtepi64_storeu_epi16(void * __P, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.mem.128
  return _mm_mask_cvtepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm256_cvtepi64_epi16(__m256i __A) {
  // CHECK-LABEL: @test_mm256_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.256
  return _mm256_cvtepi64_epi16(__A); 
}

__m128i test_mm256_mask_cvtepi64_epi16(__m128i __O, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.256
  return _mm256_mask_cvtepi64_epi16(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi64_epi16(__mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.256
  return _mm256_maskz_cvtepi64_epi16(__M, __A); 
}

void test_mm256_mask_cvtepi64_storeu_epi16(void * __P, __mmask8 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.mem.256
  return _mm256_mask_cvtepi64_storeu_epi16(__P, __M, __A); 
}

__m128 test_mm256_extractf32x4_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_extractf32x4_ps
  // CHECK: @llvm.x86.avx512.mask.vextractf32x4
  return _mm256_extractf32x4_ps(__A, 1); 
}

__m128 test_mm256_mask_extractf32x4_ps(__m128 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_extractf32x4_ps
  // CHECK: @llvm.x86.avx512.mask.vextractf32x4
  return _mm256_mask_extractf32x4_ps(__W, __U, __A, 1); 
}

__m128 test_mm256_maskz_extractf32x4_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_extractf32x4_ps
  // CHECK: @llvm.x86.avx512.mask.vextractf32x4
  return _mm256_maskz_extractf32x4_ps(__U, __A, 1); 
}

__m128i test_mm256_extracti32x4_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm256_extracti32x4_epi32
  // CHECK: @llvm.x86.avx512.mask.vextracti32x4
  return _mm256_extracti32x4_epi32(__A, 1); 
}

__m128i test_mm256_mask_extracti32x4_epi32(__m128i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_extracti32x4_epi32
  // CHECK: @llvm.x86.avx512.mask.vextracti32x4
  return _mm256_mask_extracti32x4_epi32(__W, __U, __A, 1); 
}

__m128i test_mm256_maskz_extracti32x4_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_extracti32x4_epi32
  // CHECK: @llvm.x86.avx512.mask.vextracti32x4
  return _mm256_maskz_extracti32x4_epi32(__U, __A, 1); 
}

__m256 test_mm256_insertf32x4(__m256 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm256_insertf32x4
  // CHECK: @llvm.x86.avx512.mask.insertf32x4
  return _mm256_insertf32x4(__A, __B, 1); 
}

__m256 test_mm256_mask_insertf32x4(__m256 __W, __mmask8 __U, __m256 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm256_mask_insertf32x4
  // CHECK: @llvm.x86.avx512.mask.insertf32x4
  return _mm256_mask_insertf32x4(__W, __U, __A, __B, 1); 
}

__m256 test_mm256_maskz_insertf32x4(__mmask8 __U, __m256 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm256_maskz_insertf32x4
  // CHECK: @llvm.x86.avx512.mask.insertf32x4
  return _mm256_maskz_insertf32x4(__U, __A, __B, 1); 
}

__m256i test_mm256_inserti32x4(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_inserti32x4
  // CHECK: @llvm.x86.avx512.mask.inserti32x4
  return _mm256_inserti32x4(__A, __B, 1); 
}

__m256i test_mm256_mask_inserti32x4(__m256i __W, __mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_inserti32x4
  // CHECK: @llvm.x86.avx512.mask.inserti32x4
  return _mm256_mask_inserti32x4(__W, __U, __A, __B, 1); 
}

__m256i test_mm256_maskz_inserti32x4(__mmask8 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_inserti32x4
  // CHECK: @llvm.x86.avx512.mask.inserti32x4
  return _mm256_maskz_inserti32x4(__U, __A, __B, 1); 
}

__m128d test_mm_getmant_pd(__m128d __A) {
  // CHECK-LABEL: @test_mm_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.128
  return _mm_getmant_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128d test_mm_mask_getmant_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.128
  return _mm_mask_getmant_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128d test_mm_maskz_getmant_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.128
  return _mm_maskz_getmant_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256d test_mm256_getmant_pd(__m256d __A) {
  // CHECK-LABEL: @test_mm256_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.256
  return _mm256_getmant_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256d test_mm256_mask_getmant_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.256
  return _mm256_mask_getmant_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256d test_mm256_maskz_getmant_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.256
  return _mm256_maskz_getmant_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128 test_mm_getmant_ps(__m128 __A) {
  // CHECK-LABEL: @test_mm_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.128
  return _mm_getmant_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128 test_mm_mask_getmant_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.128
  return _mm_mask_getmant_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128 test_mm_maskz_getmant_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.128
  return _mm_maskz_getmant_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256 test_mm256_getmant_ps(__m256 __A) {
  // CHECK-LABEL: @test_mm256_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.256
  return _mm256_getmant_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256 test_mm256_mask_getmant_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.256
  return _mm256_mask_getmant_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m256 test_mm256_maskz_getmant_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.256
  return _mm256_maskz_getmant_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m128d test_mm_mmask_i64gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_pd
  // CHECK: @llvm.x86.avx512.gather3div2.df
  return _mm_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_epi64
  // CHECK: @llvm.x86.avx512.gather3div2.di
  return _mm_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mmask_i64gather_pd(__m256d __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_pd
  // CHECK: @llvm.x86.avx512.gather3div4.df
  return _mm256_mmask_i64gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mmask_i64gather_epi64(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_epi64
  // CHECK: @llvm.x86.avx512.gather3div4.di
  return _mm256_mmask_i64gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_ps
  // CHECK: @llvm.x86.avx512.gather3div4.sf
  return _mm_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mmask_i64gather_epi32
  // CHECK: @llvm.x86.avx512.gather3div4.si
  return _mm_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm256_mmask_i64gather_ps(__m128 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_ps
  // CHECK: @llvm.x86.avx512.gather3div8.sf
  return _mm256_mmask_i64gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm256_mmask_i64gather_epi32(__m128i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mmask_i64gather_epi32
  // CHECK: @llvm.x86.avx512.gather3div8.si
  return _mm256_mmask_i64gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m128d test_mm_mask_i32gather_pd(__m128d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_pd
  // CHECK: @llvm.x86.avx512.gather3siv2.df
  return _mm_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi64(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_epi64
  // CHECK: @llvm.x86.avx512.gather3siv2.di
  return _mm_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_mask_i32gather_pd(__m256d __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_pd
  // CHECK: @llvm.x86.avx512.gather3siv4.df
  return _mm256_mmask_i32gather_pd(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi64(__m256i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_epi64
  // CHECK: @llvm.x86.avx512.gather3siv4.di
  return _mm256_mmask_i32gather_epi64(__v1_old, __mask, __index, __addr, 2); 
}

__m128 test_mm_mask_i32gather_ps(__m128 __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_ps
  // CHECK: @llvm.x86.avx512.gather3siv4.sf
  return _mm_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m128i test_mm_mask_i32gather_epi32(__m128i __v1_old, __mmask8 __mask, __m128i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm_mask_i32gather_epi32
  // CHECK: @llvm.x86.avx512.gather3siv4.si
  return _mm_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m256 test_mm256_mask_i32gather_ps(__m256 __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_ps
  // CHECK: @llvm.x86.avx512.gather3siv8.sf
  return _mm256_mmask_i32gather_ps(__v1_old, __mask, __index, __addr, 2); 
}

__m256i test_mm256_mask_i32gather_epi32(__m256i __v1_old, __mmask8 __mask, __m256i __index, void const *__addr) {
  // CHECK-LABEL: @test_mm256_mask_i32gather_epi32
  // CHECK: @llvm.x86.avx512.gather3siv8.si
  return _mm256_mmask_i32gather_epi32(__v1_old, __mask, __index, __addr, 2); 
}

__m256d test_mm256_permutex_pd(__m256d __X) {
  // CHECK-LABEL: @test_mm256_permutex_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> undef, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  return _mm256_permutex_pd(__X, 3);
}

__m256d test_mm256_mask_permutex_pd(__m256d __W, __mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_mask_permutex_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_permutex_pd(__W, __U, __X, 1);
}

__m256d test_mm256_maskz_permutex_pd(__mmask8 __U, __m256d __X) {
  // CHECK-LABEL: @test_mm256_maskz_permutex_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> undef, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_permutex_pd(__U, __X, 1);
}

__m256i test_mm256_permutex_epi64(__m256i __X) {
  // CHECK-LABEL: @test_mm256_permutex_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  return _mm256_permutex_epi64(__X, 3);
}

__m256i test_mm256_mask_permutex_epi64(__m256i __W, __mmask8 __M, __m256i __X) {
  // CHECK-LABEL: @test_mm256_mask_permutex_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_permutex_epi64(__W, __M, __X, 3);
}

__m256i test_mm256_maskz_permutex_epi64(__mmask8 __M, __m256i __X) {
  // CHECK-LABEL: @test_mm256_maskz_permutex_epi64
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> undef, <4 x i32> <i32 3, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_permutex_epi64(__M, __X, 3);
}

__m256d test_mm256_permutexvar_pd(__m256i __X, __m256d __Y) {
  // CHECK-LABEL: @test_mm256_permutexvar_pd
  // CHECK: @llvm.x86.avx512.mask.permvar.df.256
  return _mm256_permutexvar_pd(__X, __Y);
}

__m256d test_mm256_mask_permutexvar_pd(__m256d __W, __mmask8 __U, __m256i __X, __m256d __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_pd
  // CHECK: @llvm.x86.avx512.mask.permvar.df.256
  return _mm256_mask_permutexvar_pd(__W, __U, __X, __Y);
}

__m256d test_mm256_maskz_permutexvar_pd(__mmask8 __U, __m256i __X, __m256d __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_pd
  // CHECK: @llvm.x86.avx512.mask.permvar.df.256
  return _mm256_maskz_permutexvar_pd(__U, __X, __Y);
}

__m256i test_mm256_maskz_permutexvar_epi64(__mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.mask.permvar.di.256
  return _mm256_maskz_permutexvar_epi64(__M, __X, __Y);
}

__m256i test_mm256_mask_permutexvar_epi64(__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_epi64
  // CHECK: @llvm.x86.avx512.mask.permvar.di.256
  return _mm256_mask_permutexvar_epi64(__W, __M, __X, __Y);
}

__m256 test_mm256_mask_permutexvar_ps(__m256 __W, __mmask8 __U, __m256i __X, __m256 __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_ps
  // CHECK: @llvm.x86.avx512.mask.permvar.sf.256
  return _mm256_mask_permutexvar_ps(__W, __U, __X, __Y);
}

__m256 test_mm256_maskz_permutexvar_ps(__mmask8 __U, __m256i __X, __m256 __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_ps
  // CHECK: @llvm.x86.avx512.mask.permvar.sf.256
  return _mm256_maskz_permutexvar_ps(__U, __X, __Y);
}

__m256 test_mm256_permutexvar_ps(__m256i __X, __m256 __Y) {
  // CHECK-LABEL: @test_mm256_permutexvar_ps
  // CHECK: @llvm.x86.avx512.mask.permvar.sf.256
  return _mm256_permutexvar_ps( __X, __Y);
}

__m256i test_mm256_maskz_permutexvar_epi32(__mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_epi32
  // CHECK: @llvm.x86.avx512.mask.permvar.si.256
  return _mm256_maskz_permutexvar_epi32(__M, __X, __Y);
}

__m256i test_mm256_permutexvar_epi32(__m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_permutexvar_epi32
  // CHECK: @llvm.x86.avx512.mask.permvar.si.256
  return _mm256_permutexvar_epi32(__X, __Y);
}

__m256i test_mm256_mask_permutexvar_epi32(__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_epi32
  // CHECK: @llvm.x86.avx512.mask.permvar.si.256
  return _mm256_mask_permutexvar_epi32(__W, __M, __X, __Y);
}

__m128i test_mm_alignr_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.128
  return _mm_alignr_epi32(__A, __B, 1);
}

__m128i test_mm_mask_alignr_epi32(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.128
  return _mm_mask_alignr_epi32(__W, __U, __A, __B, 1);
}

__m128i test_mm_maskz_alignr_epi32(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.128
  return _mm_maskz_alignr_epi32(__U, __A, __B, 1);
}

__m256i test_mm256_alignr_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.256
  return _mm256_alignr_epi32(__A, __B, 1);
}

__m256i test_mm256_mask_alignr_epi32(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.256
  return _mm256_mask_alignr_epi32(__W, __U, __A, __B, 1);
}

__m256i test_mm256_maskz_alignr_epi32(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.256
  return _mm256_maskz_alignr_epi32(__U, __A, __B, 1);
}

__m128i test_mm_alignr_epi64(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.128
  return _mm_alignr_epi64(__A, __B, 1);
}

__m128i test_mm_mask_alignr_epi64(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.128
  return _mm_mask_alignr_epi64(__W, __U, __A, __B, 1);
}

__m128i test_mm_maskz_alignr_epi64(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.128
  return _mm_maskz_alignr_epi64(__U, __A, __B, 1);
}

__m256i test_mm256_alignr_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.256
  return _mm256_alignr_epi64(__A, __B, 1);
}

__m256i test_mm256_mask_alignr_epi64(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.256
  return _mm256_mask_alignr_epi64(__W, __U, __A, __B, 1);
}

__m256i test_mm256_maskz_alignr_epi64(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.256
  return _mm256_maskz_alignr_epi64(__U, __A, __B, 1);
}

__m128 test_mm_mask_movehdup_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_movehdup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_movehdup_ps(__W, __U, __A);
}

__m128 test_mm_maskz_movehdup_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_movehdup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_movehdup_ps(__U, __A);
}

__m256 test_mm256_mask_movehdup_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_movehdup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_movehdup_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_movehdup_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_movehdup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 3, i32 3, i32 5, i32 5, i32 7, i32 7>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_movehdup_ps(__U, __A);
}

__m128 test_mm_mask_moveldup_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_moveldup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_moveldup_ps(__W, __U, __A);
}

__m128 test_mm_maskz_moveldup_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_moveldup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  // CHECK: select <4 x i1> %{{.*}} <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_moveldup_ps(__U, __A);
}

__m256 test_mm256_mask_moveldup_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_moveldup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_moveldup_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_moveldup_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_moveldup_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 0, i32 2, i32 2, i32 4, i32 4, i32 6, i32 6>
  // CHECK: select <8 x i1> %{{.*}} <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_moveldup_ps(__U, __A);
}

__m128i test_mm_mask_shuffle_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_shuffle_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_shuffle_epi32(__W, __U, __A, 1);
}

__m128i test_mm_maskz_shuffle_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 2, i32 0, i32 0, i32 0>
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_shuffle_epi32(__U, __A, 2);
}

__m256i test_mm256_mask_shuffle_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_shuffle_epi32(__W, __U, __A, 2);
}

__m256i test_mm256_maskz_shuffle_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_epi32
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 0, i32 0, i32 0, i32 6, i32 4, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_shuffle_epi32(__U, __A, 2);
}

__m128d test_mm_mask_mov_pd(__m128d __W, __mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_mask_mov_pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_mask_mov_pd(__W, __U, __A);
}

__m128d test_mm_maskz_mov_pd(__mmask8 __U, __m128d __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_pd
  // CHECK: select <2 x i1> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_maskz_mov_pd(__U, __A);
}

__m256d test_mm256_mask_mov_pd(__m256d __W, __mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_pd
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_mask_mov_pd(__W, __U, __A);
}

__m256d test_mm256_maskz_mov_pd(__mmask8 __U, __m256d __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_pd
  // CHECK: select <4 x i1> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_maskz_mov_pd(__U, __A);
}

__m128 test_mm_mask_mov_ps(__m128 __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_mov_ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_mask_mov_ps(__W, __U, __A);
}

__m128 test_mm_maskz_mov_ps(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_ps
  // CHECK: select <4 x i1> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_maskz_mov_ps(__U, __A);
}

__m256 test_mm256_mask_mov_ps(__m256 __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_mask_mov_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_mov_ps(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_ps
  // CHECK: select <8 x i1> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_maskz_mov_ps(__U, __A);
}

__m128 test_mm_mask_cvtph_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtph_ps
  // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.128
  return _mm_mask_cvtph_ps(__W, __U, __A);
}

__m128 test_mm_maskz_cvtph_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtph_ps
  // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.128
  return _mm_maskz_cvtph_ps(__U, __A);
}

__m256 test_mm256_mask_cvtph_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtph_ps
  // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.256
  return _mm256_mask_cvtph_ps(__W, __U, __A);
}

__m256 test_mm256_maskz_cvtph_ps(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtph_ps
  // CHECK: @llvm.x86.avx512.mask.vcvtph2ps.256
  return _mm256_maskz_cvtph_ps(__U, __A);
}

__m128i test_mm_mask_cvtps_ph(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_mask_cvtps_ph(__W, __U, __A);
}

__m128i test_mm_maskz_cvtps_ph(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_maskz_cvtps_ph(__U, __A);
}

__m128i test_mm256_mask_cvtps_ph(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_mask_cvtps_ph(__W, __U, __A);
}

__m128i test_mm256_maskz_cvtps_ph(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_maskz_cvtps_ph(__U, __A);
}

__m128i test_mm_mask_cvt_roundps_ph(__m128i __W, __mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_mask_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_CUR_DIRECTION);
}

__m128i test_mm_maskz_cvt_roundps_ph(__mmask8 __U, __m128 __A) {
  // CHECK-LABEL: @test_mm_maskz_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.128
  return _mm_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_CUR_DIRECTION);
}

__m128i test_mm256_mask_cvt_roundps_ph(__m128i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_mask_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_mask_cvt_roundps_ph(__W, __U, __A, _MM_FROUND_CUR_DIRECTION);
}

__m128i test_mm256_maskz_cvt_roundps_ph(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvt_roundps_ph
  // CHECK: @llvm.x86.avx512.mask.vcvtps2ph.256
  return _mm256_maskz_cvt_roundps_ph(__U, __A, _MM_FROUND_CUR_DIRECTION);
}

__mmask8 test_mm_cmpeq_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi32_mask
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi64_mask
  // CHECK: icmp eq <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm_cmpgt_epi32_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi32_mask
  // CHECK: icmp sgt <4 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi32_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi32_mask
  // CHECK: icmp sgt <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi64_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi64_mask
  // CHECK: icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <2 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi64_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi64_mask
  // CHECK: icmp sgt <2 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epi64_mask(__a, __b);
}

__mmask8 test_mm256_cmpeq_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpeq_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi32_mask
  // CHECK: icmp eq <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_mask_cmpeq_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpeq_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi64_mask
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpeq_epi64_mask(__a, __b);
}

__mmask8 test_mm256_cmpgt_epi32_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi32_mask
  // CHECK: icmp sgt <8 x i32> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epi32_mask(__a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi32_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi32_mask
  // CHECK: icmp sgt <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm256_mask_cmpgt_epi64_mask(__mmask8 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi64_mask
  // CHECK: icmp sgt <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: and <4 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm256_cmpgt_epi64_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi64_mask
  // CHECK: icmp sgt <4 x i64> %{{.*}}, %{{.*}}
  return (__mmask8)_mm256_cmpgt_epi64_mask(__a, __b);
}
