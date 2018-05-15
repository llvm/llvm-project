// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__mmask32 test_mm256_cmpeq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpeq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpeq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpeq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi8_mask
  // CHECK: icmp sgt <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpgt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi8_mask
  // CHECK: icmp sgt <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi8_mask
  // CHECK: icmp sgt <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpgt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi8_mask
  // CHECK: icmp sgt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epi16_mask
  // CHECK: icmp sgt <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpgt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epi16_mask
  // CHECK: icmp sgt <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epi16_mask
  // CHECK: icmp sgt <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epi16_mask
  // CHECK: icmp sgt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpeq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpeq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpeq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpeq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu8_mask
  // CHECK: icmp ugt <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpgt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu8_mask
  // CHECK: icmp ugt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpgt_epu16_mask
  // CHECK: icmp ugt <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpgt_epu16_mask
  // CHECK: icmp ugt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu8_mask
  // CHECK: icmp ugt <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpgt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu8_mask
  // CHECK: icmp ugt <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpgt_epu16_mask
  // CHECK: icmp ugt <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpgt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpgt_epu16_mask
  // CHECK: icmp ugt <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi8_mask
  // CHECK: icmp sge <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpge_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi8_mask
  // CHECK: icmp sge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu8_mask
  // CHECK: icmp uge <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpge_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu8_mask
  // CHECK: icmp uge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epi16_mask
  // CHECK: icmp sge <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epi16_mask
  // CHECK: icmp sge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpge_epu16_mask
  // CHECK: icmp uge <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpge_epu16_mask
  // CHECK: icmp uge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi8_mask
  // CHECK: icmp sge <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpge_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi8_mask
  // CHECK: icmp sge <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu8_mask
  // CHECK: icmp uge <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpge_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu8_mask
  // CHECK: icmp uge <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epi16_mask
  // CHECK: icmp sge <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpge_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epi16_mask
  // CHECK: icmp sge <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpge_epu16_mask
  // CHECK: icmp uge <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpge_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpge_epu16_mask
  // CHECK: icmp uge <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi8_mask
  // CHECK: icmp sle <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmple_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi8_mask
  // CHECK: icmp sle <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu8_mask
  // CHECK: icmp ule <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmple_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu8_mask
  // CHECK: icmp ule <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epi16_mask
  // CHECK: icmp sle <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epi16_mask
  // CHECK: icmp sle <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmple_epu16_mask
  // CHECK: icmp ule <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmple_epu16_mask
  // CHECK: icmp ule <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi8_mask
  // CHECK: icmp sle <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmple_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi8_mask
  // CHECK: icmp sle <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu8_mask
  // CHECK: icmp ule <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmple_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu8_mask
  // CHECK: icmp ule <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epi16_mask
  // CHECK: icmp sle <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmple_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epi16_mask
  // CHECK: icmp sle <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmple_epu16_mask
  // CHECK: icmp ule <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmple_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmple_epu16_mask
  // CHECK: icmp ule <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi8_mask
  // CHECK: icmp slt <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmplt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi8_mask
  // CHECK: icmp slt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu8_mask
  // CHECK: icmp ult <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmplt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu8_mask
  // CHECK: icmp ult <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epi16_mask
  // CHECK: icmp slt <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epi16_mask
  // CHECK: icmp slt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmplt_epu16_mask
  // CHECK: icmp ult <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmplt_epu16_mask
  // CHECK: icmp ult <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi8_mask
  // CHECK: icmp slt <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmplt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi8_mask
  // CHECK: icmp slt <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu8_mask
  // CHECK: icmp ult <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmplt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu8_mask
  // CHECK: icmp ult <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epi16_mask
  // CHECK: icmp slt <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmplt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epi16_mask
  // CHECK: icmp slt <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmplt_epu16_mask
  // CHECK: icmp ult <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmplt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmplt_epu16_mask
  // CHECK: icmp ult <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpneq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpneq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epi16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epi16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpneq_epu16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpneq_epu16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpneq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpneq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epi16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpneq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epi16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpneq_epu16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpneq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpneq_epu16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmp_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmp_epi8_mask(__a, __b, 0);
}

__mmask16 test_mm_mask_cmp_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmp_epi8_mask(__u, __a, __b, 0);
}

__mmask16 test_mm_cmp_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmp_epu8_mask(__a, __b, 0);
}

__mmask16 test_mm_mask_cmp_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmp_epu8_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi16_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi16_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmp_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu16_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmp_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu16_mask(__u, __a, __b, 0);
}

__mmask32 test_mm256_cmp_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmp_epi8_mask(__a, __b, 0);
}

__mmask32 test_mm256_mask_cmp_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmp_epi8_mask(__u, __a, __b, 0);
}

__mmask32 test_mm256_cmp_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmp_epu8_mask(__a, __b, 0);
}

__mmask32 test_mm256_mask_cmp_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmp_epu8_mask(__u, __a, __b, 0);
}

__mmask16 test_mm256_cmp_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmp_epi16_mask(__a, __b, 0);
}

__mmask16 test_mm256_mask_cmp_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmp_epi16_mask(__u, __a, __b, 0);
}

__mmask16 test_mm256_cmp_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmp_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmp_epu16_mask(__a, __b, 0);
}

__mmask16 test_mm256_mask_cmp_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmp_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmp_epu16_mask(__u, __a, __b, 0);
}


__m256i test_mm256_mask_add_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B){
  //CHECK-LABEL: @test_mm256_mask_add_epi8
  //CHECK: add <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_add_epi8(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_add_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi8
  //CHECK: add <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_add_epi8(__U , __A, __B);
}
__m256i test_mm256_mask_add_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_add_epi16
  //CHECK: add <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_add_epi16(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_add_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_add_epi16
  //CHECK: add <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_add_epi16(__U , __A, __B);
}

__m256i test_mm256_mask_sub_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi8
  //CHECK: sub <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_sub_epi8(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_sub_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi8
  //CHECK: sub <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_sub_epi8(__U , __A, __B);
}

__m256i test_mm256_mask_sub_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_sub_epi16
  //CHECK: sub <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sub_epi16(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_sub_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_sub_epi16
  //CHECK: sub <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sub_epi16(__U , __A, __B);
}

__m128i test_mm_mask_add_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi8
  //CHECK: add <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_add_epi8(__W, __U , __A, __B);
}

__m128i test_mm_maskz_add_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi8
  //CHECK: add <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_add_epi8(__U , __A, __B);
}

__m128i test_mm_mask_add_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_add_epi16
  //CHECK: add <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_add_epi16(__W, __U , __A, __B);
}

__m128i test_mm_maskz_add_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_add_epi16
  //CHECK: add <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_add_epi16(__U , __A, __B);
}

__m128i test_mm_mask_sub_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi8
  //CHECK: sub <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_sub_epi8(__W, __U , __A, __B);
}

__m128i test_mm_maskz_sub_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi8
  //CHECK: sub <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_sub_epi8(__U , __A, __B);
}

__m128i test_mm_mask_sub_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_sub_epi16
  //CHECK: sub <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sub_epi16(__W, __U , __A, __B);
}

__m128i test_mm_maskz_sub_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_sub_epi16
  //CHECK: sub <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sub_epi16(__U , __A, __B);
}

__m256i test_mm256_mask_mullo_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_mask_mullo_epi16
  //CHECK: mul <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mullo_epi16(__W, __U , __A, __B);
}

__m256i test_mm256_maskz_mullo_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: @test_mm256_maskz_mullo_epi16
  //CHECK: mul <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mullo_epi16(__U , __A, __B);
}

__m128i test_mm_mask_mullo_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_mask_mullo_epi16
  //CHECK: mul <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mullo_epi16(__W, __U , __A, __B);
}

__m128i test_mm_maskz_mullo_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: @test_mm_maskz_mullo_epi16
  //CHECK: mul <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mullo_epi16(__U , __A, __B);
}


__m128i test_mm_mask_blend_epi8(__mmask16 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_blend_epi8(__U,__A,__W); 
}
__m256i test_mm256_mask_blend_epi8(__mmask32 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_blend_epi8(__U,__A,__W); 
}

__m128i test_mm_mask_blend_epi16(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: @test_mm_mask_blend_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_blend_epi16(__U,__A,__W); 
}

__m256i test_mm256_mask_blend_epi16(__mmask16 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: @test_mm256_mask_blend_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_blend_epi16(__U,__A,__W); 
}

__m128i test_mm_mask_abs_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi8
  // CHECK: [[SUB:%.*]] = sub <16 x i8> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <16 x i8> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[A]], <16 x i8> [[SUB]]
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[SEL]], <16 x i8> %{{.*}}
  return _mm_mask_abs_epi8(__W,__U,__A); 
}

__m128i test_mm_maskz_abs_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi8
  // CHECK: [[SUB:%.*]] = sub <16 x i8> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <16 x i8> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[A]], <16 x i8> [[SUB]]
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[SEL]], <16 x i8> %{{.*}}
  return _mm_maskz_abs_epi8(__U,__A); 
}

__m256i test_mm256_mask_abs_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi8
  // CHECK: [[SUB:%.*]] = sub <32 x i8> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <32 x i8> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[A]], <32 x i8> [[SUB]]
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[SEL]], <32 x i8> %{{.*}}
  return _mm256_mask_abs_epi8(__W,__U,__A); 
}

__m256i test_mm256_maskz_abs_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi8
  // CHECK: [[SUB:%.*]] = sub <32 x i8> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <32 x i8> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[A]], <32 x i8> [[SUB]]
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[SEL]], <32 x i8> %{{.*}}
  return _mm256_maskz_abs_epi8(__U,__A); 
}

__m128i test_mm_mask_abs_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_abs_epi16
  // CHECK: [[SUB:%.*]] = sub <8 x i16> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <8 x i16> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[A]], <8 x i16> [[SUB]]
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> [[SEL]], <8 x i16> %{{.*}}
  return _mm_mask_abs_epi16(__W,__U,__A); 
}

__m128i test_mm_maskz_abs_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_abs_epi16
  // CHECK: [[SUB:%.*]] = sub <8 x i16> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <8 x i16> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[A]], <8 x i16> [[SUB]]
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> [[SEL]], <8 x i16> %{{.*}}
  return _mm_maskz_abs_epi16(__U,__A); 
}

__m256i test_mm256_mask_abs_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_abs_epi16
  // CHECK: [[SUB:%.*]] = sub <16 x i16> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <16 x i16> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[A]], <16 x i16> [[SUB]]
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> [[SEL]], <16 x i16> %{{.*}}
  return _mm256_mask_abs_epi16(__W,__U,__A); 
}

__m256i test_mm256_maskz_abs_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_abs_epi16
  // CHECK: [[SUB:%.*]] = sub <16 x i16> zeroinitializer, [[A:%.*]]
  // CHECK: [[CMP:%.*]] = icmp sgt <16 x i16> [[A]], zeroinitializer
  // CHECK: [[SEL:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[A]], <16 x i16> [[SUB]]
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> [[SEL]], <16 x i16> %{{.*}}
  return _mm256_maskz_abs_epi16(__U,__A); 
}

__m128i test_mm_maskz_packs_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packs_epi32
  // CHECK: @llvm.x86.sse2.packssdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_packs_epi32(__M,__A,__B); 
}
__m128i test_mm_mask_packs_epi32(__m128i __W, __mmask16 __M, __m128i __A,          __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packs_epi32
  // CHECK: @llvm.x86.sse2.packssdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_packs_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_packs_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packs_epi32
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_packs_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_packs_epi32(__m256i __W, __mmask16 __M, __m256i __A,       __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packs_epi32
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_packs_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_packs_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packs_epi16
  // CHECK: @llvm.x86.sse2.packsswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_packs_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_packs_epi16(__m128i __W, __mmask16 __M, __m128i __A,          __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packs_epi16
  // CHECK: @llvm.x86.sse2.packsswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_packs_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_packs_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packs_epi16
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_packs_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_packs_epi16(__m256i __W, __mmask32 __M, __m256i __A,       __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packs_epi16
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_packs_epi16(__W,__M,__A,__B); 
}

__m128i test_mm_mask_packus_epi32(__m128i __W, __mmask16 __M, __m128i __A,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packus_epi32
  // CHECK: @llvm.x86.sse41.packusdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_packus_epi32(__W,__M,__A,__B); 
}

__m128i test_mm_maskz_packus_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packus_epi32
  // CHECK: @llvm.x86.sse41.packusdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_packus_epi32(__M,__A,__B); 
}

__m256i test_mm256_maskz_packus_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packus_epi32
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_packus_epi32(__M,__A,__B); 
}

__m256i test_mm256_mask_packus_epi32(__m256i __W, __mmask16 __M, __m256i __A,        __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packus_epi32
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_packus_epi32(__W,__M,__A,__B); 
}

__m128i test_mm_maskz_packus_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_packus_epi16
  // CHECK: @llvm.x86.sse2.packuswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_packus_epi16(__M,__A,__B); 
}

__m128i test_mm_mask_packus_epi16(__m128i __W, __mmask16 __M, __m128i __A,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_packus_epi16
  // CHECK: @llvm.x86.sse2.packuswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_packus_epi16(__W,__M,__A,__B); 
}

__m256i test_mm256_maskz_packus_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_packus_epi16
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_packus_epi16(__M,__A,__B); 
}

__m256i test_mm256_mask_packus_epi16(__m256i __W, __mmask32 __M, __m256i __A,        __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_packus_epi16
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_packus_epi16(__W,__M,__A,__B); 
}

__m128i test_mm_mask_adds_epi8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epi8
  // CHECK: @llvm.x86.sse2.padds.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_adds_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epi8
  // CHECK: @llvm.x86.sse2.padds.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_adds_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epi8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epi8
  // CHECK: @llvm.x86.avx2.padds.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_adds_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epi8
  // CHECK: @llvm.x86.avx2.padds.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_adds_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epi16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epi16
  // CHECK: @llvm.x86.sse2.padds.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_adds_epi16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epi16
  // CHECK: @llvm.x86.sse2.padds.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_adds_epi16(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epi16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epi16
  // CHECK: @llvm.x86.avx2.padds.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_adds_epi16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epi16
  // CHECK: @llvm.x86.avx2.padds.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_adds_epi16(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epu8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epu8
  // CHECK: @llvm.x86.sse2.paddus.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_adds_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epu8
  // CHECK: @llvm.x86.sse2.paddus.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_adds_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epu8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epu8
  // CHECK: @llvm.x86.avx2.paddus.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_adds_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epu8
  // CHECK: @llvm.x86.avx2.paddus.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_adds_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epu16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_adds_epu16
  // CHECK: @llvm.x86.sse2.paddus.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_adds_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_adds_epu16
  // CHECK: @llvm.x86.sse2.paddus.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_adds_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epu16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_adds_epu16
  // CHECK: @llvm.x86.avx2.paddus.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_adds_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_adds_epu16
  // CHECK: @llvm.x86.avx2.paddus.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_adds_epu16(__U,__A,__B); 
}
__m128i test_mm_mask_avg_epu8(__m128i __W, __mmask16 __U, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_avg_epu8
  // CHECK-NOT: @llvm.x86.sse2.pavg.b
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: add <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: add <16 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: lshr <16 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_avg_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_avg_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_avg_epu8
  // CHECK-NOT: @llvm.x86.sse2.pavg.b
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: add <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: add <16 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: lshr <16 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_avg_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_avg_epu8(__m256i __W, __mmask32 __U, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_avg_epu8
  // CHECK-NOT: @llvm.x86.avx2.pavg.b
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: add <32 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: lshr <32 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_avg_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_avg_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_avg_epu8
  // CHECK-NOT: @llvm.x86.avx2.pavg.b
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: zext <32 x i8> %{{.*}} to <32 x i16>
  // CHECK: add <32 x i16> %{{.*}}, %{{.*}}
  // CHECK: add <32 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: lshr <32 x i16> %{{.*}}, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // CHECK: trunc <32 x i16> %{{.*}} to <32 x i8>
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_avg_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_avg_epu16(__m128i __W, __mmask8 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_avg_epu16
  // CHECK-NOT: @llvm.x86.sse2.pavg.w
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: add <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: add <8 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: lshr <8 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: trunc <8 x i32> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_avg_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_avg_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_avg_epu16
  // CHECK-NOT: @llvm.x86.sse2.pavg.w
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: zext <8 x i16> %{{.*}} to <8 x i32>
  // CHECK: add <8 x i32> %{{.*}}, %{{.*}}
  // CHECK: add <8 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: lshr <8 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: trunc <8 x i32> %{{.*}} to <8 x i16>
  // CHECK: store <2 x i64> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_avg_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_avg_epu16(__m256i __W, __mmask16 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_avg_epu16
  // CHECK-NOT: @llvm.x86.avx2.pavg.w
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: add <16 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: lshr <16 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_avg_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_avg_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_avg_epu16
  // CHECK-NOT: @llvm.x86.avx2.pavg.w
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: zext <16 x i16> %{{.*}} to <16 x i32>
  // CHECK: add <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: add <16 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: lshr <16 x i32> %{{.*}}, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  // CHECK: trunc <16 x i32> %{{.*}} to <16 x i16>
  // CHECK: store <4 x i64> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_avg_epu16(__U,__A,__B); 
}
__m128i test_mm_maskz_max_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi8
  // CHECK:       [[CMP:%.*]] = icmp sgt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_max_epi8(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi8
  // CHECK:       [[CMP:%.*]] = icmp sgt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_max_epi8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi8
  // CHECK:       [[CMP:%.*]] = icmp sgt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_max_epi8(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi8
  // CHECK:       [[CMP:%.*]] = icmp sgt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_max_epi8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epi16
  // CHECK:       [[CMP:%.*]] = icmp sgt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_max_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_max_epi16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epi16
  // CHECK:       [[CMP:%.*]] = icmp sgt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_max_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epi16
  // CHECK:       [[CMP:%.*]] = icmp sgt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_max_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epi16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epi16
  // CHECK:       [[CMP:%.*]] = icmp sgt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_max_epi16(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epu8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu8
  // CHECK:       [[CMP:%.*]] = icmp ugt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_max_epu8(__M,__A,__B); 
}
__m128i test_mm_mask_max_epu8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu8
  // CHECK:       [[CMP:%.*]] = icmp ugt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_max_epu8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu8
  // CHECK:       [[CMP:%.*]] = icmp ugt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_max_epu8(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epu8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu8
  // CHECK:       [[CMP:%.*]] = icmp ugt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_max_epu8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_max_epu16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_max_epu16
  // CHECK:       [[CMP:%.*]] = icmp ugt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_max_epu16(__M,__A,__B); 
}
__m128i test_mm_mask_max_epu16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_max_epu16
  // CHECK:       [[CMP:%.*]] = icmp ugt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_max_epu16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_max_epu16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_max_epu16
  // CHECK:       [[CMP:%.*]] = icmp ugt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_max_epu16(__M,__A,__B); 
}
__m256i test_mm256_mask_max_epu16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_max_epu16
  // CHECK:       [[CMP:%.*]] = icmp ugt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_max_epu16(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi8
  // CHECK:       [[CMP:%.*]] = icmp slt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_min_epi8(__M,__A,__B); 
}
__m128i test_mm_mask_min_epi8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi8
  // CHECK:       [[CMP:%.*]] = icmp slt <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_min_epi8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi8
  // CHECK:       [[CMP:%.*]] = icmp slt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_min_epi8(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epi8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi8
  // CHECK:       [[CMP:%.*]] = icmp slt <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_min_epi8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epi16
  // CHECK:       [[CMP:%.*]] = icmp slt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_min_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_min_epi16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epi16
  // CHECK:       [[CMP:%.*]] = icmp slt <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_min_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epi16
  // CHECK:       [[CMP:%.*]] = icmp slt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_min_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epi16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epi16
  // CHECK:       [[CMP:%.*]] = icmp slt <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_min_epi16(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu8
  // CHECK:       [[CMP:%.*]] = icmp ult <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_min_epu8(__M,__A,__B); 
}
__m128i test_mm_mask_min_epu8(__m128i __W, __mmask16 __M, __m128i __A,       __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu8
  // CHECK:       [[CMP:%.*]] = icmp ult <16 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i8> [[X]], <16 x i8> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_min_epu8(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu8
  // CHECK:       [[CMP:%.*]] = icmp ult <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_min_epu8(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epu8(__m256i __W, __mmask32 __M, __m256i __A,          __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu8
  // CHECK:       [[CMP:%.*]] = icmp ult <32 x i8> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <32 x i1> [[CMP]], <32 x i8> [[X]], <32 x i8> [[Y]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_min_epu8(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_min_epu16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_min_epu16
  // CHECK:       [[CMP:%.*]] = icmp ult <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_min_epu16(__M,__A,__B); 
}
__m128i test_mm_mask_min_epu16(__m128i __W, __mmask8 __M, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_min_epu16
  // CHECK:       [[CMP:%.*]] = icmp ult <8 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <8 x i1> [[CMP]], <8 x i16> [[X]], <8 x i16> [[Y]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_min_epu16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_min_epu16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_min_epu16
  // CHECK:       [[CMP:%.*]] = icmp ult <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_min_epu16(__M,__A,__B); 
}
__m256i test_mm256_mask_min_epu16(__m256i __W, __mmask16 __M, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_min_epu16
  // CHECK:       [[CMP:%.*]] = icmp ult <16 x i16> [[X:%.*]], [[Y:%.*]]
  // CHECK-NEXT:  [[RES:%.*]] = select <16 x i1> [[CMP]], <16 x i16> [[X]], <16 x i16> [[Y]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_min_epu16(__W,__M,__A,__B); 
}
__m128i test_mm_mask_shuffle_epi8(__m128i __W, __mmask16 __U, __m128i __A,           __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_shuffle_epi8
  // CHECK: @llvm.x86.ssse3.pshuf.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_shuffle_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_shuffle_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_shuffle_epi8
  // CHECK: @llvm.x86.ssse3.pshuf.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_shuffle_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_shuffle_epi8(__m256i __W, __mmask32 __U, __m256i __A,        __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_shuffle_epi8
  // CHECK: @llvm.x86.avx2.pshuf.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_shuffle_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_shuffle_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_shuffle_epi8
  // CHECK: @llvm.x86.avx2.pshuf.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_shuffle_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epi8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epi8
  // CHECK: @llvm.x86.sse2.psubs.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_subs_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epi8
  // CHECK: @llvm.x86.sse2.psubs.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_subs_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epi8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epi8
  // CHECK: @llvm.x86.avx2.psubs.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_subs_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epi8
  // CHECK: @llvm.x86.avx2.psubs.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_subs_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epi16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epi16
  // CHECK: @llvm.x86.sse2.psubs.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_subs_epi16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epi16
  // CHECK: @llvm.x86.sse2.psubs.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_subs_epi16(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epi16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epi16
  // CHECK: @llvm.x86.avx2.psubs.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_subs_epi16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epi16
  // CHECK: @llvm.x86.avx2.psubs.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_subs_epi16(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epu8(__m128i __W, __mmask16 __U, __m128i __A,        __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epu8
  // CHECK: @llvm.x86.sse2.psubus.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_subs_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epu8
  // CHECK: @llvm.x86.sse2.psubus.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_subs_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epu8(__m256i __W, __mmask32 __U, __m256i __A,           __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epu8
  // CHECK: @llvm.x86.avx2.psubus.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_subs_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epu8
  // CHECK: @llvm.x86.avx2.psubus.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_subs_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epu16(__m128i __W, __mmask8 __U, __m128i __A,         __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_subs_epu16
  // CHECK: @llvm.x86.sse2.psubus.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_subs_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_subs_epu16
  // CHECK: @llvm.x86.sse2.psubus.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_subs_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epu16(__m256i __W, __mmask16 __U, __m256i __A,      __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_subs_epu16
  // CHECK: @llvm.x86.avx2.psubus.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_subs_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_subs_epu16
  // CHECK: @llvm.x86.avx2.psubus.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
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
  // CHECK: @llvm.x86.ssse3.pmadd.ub.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_maddubs_epi16(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_maddubs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_maddubs_epi16
  // CHECK: @llvm.x86.ssse3.pmadd.ub.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_maddubs_epi16(__U, __X, __Y); 
}

__m256i test_mm256_mask_maddubs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_maddubs_epi16
  // CHECK: @llvm.x86.avx2.pmadd.ub.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_maddubs_epi16(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_maddubs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_maddubs_epi16
  // CHECK: @llvm.x86.avx2.pmadd.ub.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_maddubs_epi16(__U, __X, __Y); 
}

__m128i test_mm_mask_madd_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_madd_epi16
  // CHECK: @llvm.x86.sse2.pmadd.wd
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_madd_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_madd_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_madd_epi16
  // CHECK: @llvm.x86.sse2.pmadd.wd
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_madd_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_madd_epi16(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_madd_epi16
  // CHECK: @llvm.x86.avx2.pmadd.wd
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_madd_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_madd_epi16(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_madd_epi16
  // CHECK: @llvm.x86.avx2.pmadd.wd
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
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
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  return _mm256_cvtepi16_epi8(__A); 
}

__m128i test_mm256_mask_cvtepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi16_epi8
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm256_mask_cvtepi16_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi16_epi8
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm256_maskz_cvtepi16_epi8(__M, __A); 
}

__m128i test_mm_mask_mulhrs_epi16(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_mask_mulhrs_epi16
  // CHECK: @llvm.x86.ssse3.pmul.hr.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}

__m128i test_mm_maskz_mulhrs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: @test_mm_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.ssse3.pmul.hr.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhrs_epi16(__U, __X, __Y); 
}

__m256i test_mm256_mask_mulhrs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_mask_mulhrs_epi16
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_mulhrs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: @test_mm256_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhrs_epi16(__U, __X, __Y); 
}

__m128i test_mm_mask_mulhi_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mulhi_epu16
  // CHECK: @llvm.x86.sse2.pmulhu.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mulhi_epu16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_mulhi_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mulhi_epu16
  // CHECK: @llvm.x86.sse2.pmulhu.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhi_epu16(__U, __A, __B); 
}

__m256i test_mm256_mask_mulhi_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mulhi_epu16
  // CHECK: @llvm.x86.avx2.pmulhu.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhi_epu16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_mulhi_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mulhi_epu16
  // CHECK: @llvm.x86.avx2.pmulhu.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhi_epu16(__U, __A, __B); 
}

__m128i test_mm_mask_mulhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mulhi_epi16
  // CHECK: @llvm.x86.sse2.pmulh.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mulhi_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_mulhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mulhi_epi16
  // CHECK: @llvm.x86.sse2.pmulh.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhi_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_mulhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mulhi_epi16
  // CHECK: @llvm.x86.avx2.pmulh.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhi_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_mulhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mulhi_epi16
  // CHECK: @llvm.x86.avx2.pmulh.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhi_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_unpackhi_epi8(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpackhi_epi8(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_cvtepi8_epi16(__m128i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepi8_epi16
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepi8_epi16
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_cvtepi8_epi16(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi16(__m256i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_cvtepi8_epi16(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi16(__m128i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_cvtepu8_epi16
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_cvtepu8_epi16
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_cvtepu8_epi16(__U, __A); 
}

__m256i test_mm256_mask_cvtepu8_epi16(__m256i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_cvtepu8_epi16(__U, __A); 
}

__m256i test_mm256_sllv_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.256(
  return _mm256_sllv_epi16(__A, __B); 
}

__m256i test_mm256_mask_sllv_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sllv_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sllv_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sllv_epi16(__U, __A, __B); 
}

__m128i test_mm_sllv_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.128(
  return _mm_sllv_epi16(__A, __B); 
}

__m128i test_mm_mask_sllv_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sllv_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sllv_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sllv_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_sll_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sll_epi16
  // CHECK: @llvm.x86.sse2.psll.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sll_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sll_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sll_epi16
  // CHECK: @llvm.x86.sse2.psll.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sll_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_sll_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sll_epi16
  // CHECK: @llvm.x86.avx2.psll.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sll_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sll_epi16(__mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sll_epi16
  // CHECK: @llvm.x86.avx2.psll.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sll_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_slli_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_slli_epi16
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_slli_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_slli_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_slli_epi16
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_slli_epi16(__U, __A, 5); 
}

__m256i test_mm256_mask_slli_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_slli_epi16
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_slli_epi16(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_slli_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_slli_epi16
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_slli_epi16(__U, __A, 5); 
}

__m256i test_mm256_srlv_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.256(
  return _mm256_srlv_epi16(__A, __B); 
}

__m256i test_mm256_mask_srlv_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srlv_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srlv_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srlv_epi16(__U, __A, __B); 
}

__m128i test_mm_srlv_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.128(
  return _mm_srlv_epi16(__A, __B); 
}

__m128i test_mm_mask_srlv_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srlv_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srlv_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srlv_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_srl_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srl_epi16
  // CHECK: @llvm.x86.sse2.psrl.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srl_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srl_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srl_epi16
  // CHECK: @llvm.x86.sse2.psrl.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srl_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_srl_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_srl_epi16
  // CHECK: @llvm.x86.avx2.psrl.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srl_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srl_epi16(__mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srl_epi16
  // CHECK: @llvm.x86.avx2.psrl.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srl_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_srli_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srli_epi16
  // CHECK: @llvm.x86.sse2.psrli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srli_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_srli_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srli_epi16
  // CHECK: @llvm.x86.sse2.psrli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srli_epi16(__U, __A, 5); 
}

__m256i test_mm256_mask_srli_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srli_epi16
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srli_epi16(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_srli_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srli_epi16
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srli_epi16(__U, __A, 5); 
}

__m256i test_mm256_srav_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.256(
  return _mm256_srav_epi16(__A, __B); 
}

__m256i test_mm256_mask_srav_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srav_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srav_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srav_epi16(__U, __A, __B); 
}

__m128i test_mm_srav_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.128(
  return _mm_srav_epi16(__A, __B); 
}

__m128i test_mm_mask_srav_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srav_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srav_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srav_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_sra_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_sra_epi16
  // CHECK: @llvm.x86.sse2.psra.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sra_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sra_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_sra_epi16
  // CHECK: @llvm.x86.sse2.psra.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sra_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_sra_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_mask_sra_epi16
  // CHECK: @llvm.x86.avx2.psra.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sra_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sra_epi16(__mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_maskz_sra_epi16
  // CHECK: @llvm.x86.avx2.psra.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sra_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_srai_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_srai_epi16
  // CHECK: @llvm.x86.sse2.psrai.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srai_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_srai_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_srai_epi16
  // CHECK: @llvm.x86.sse2.psrai.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srai_epi16(__U, __A, 5); 
}

__m256i test_mm256_mask_srai_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_srai_epi16
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srai_epi16(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_srai_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_srai_epi16
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srai_epi16(__U, __A, 5); 
}

__m128i test_mm_mask_mov_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_mov_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mov_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mov_epi16(__U, __A); 
}

__m256i test_mm256_mask_mov_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mov_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mov_epi16(__U, __A); 
}

__m128i test_mm_mask_mov_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_mov_epi8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_mov_epi8(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_mov_epi8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_mov_epi8(__U, __A); 
}

__m256i test_mm256_mask_mov_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_mov_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_mov_epi8(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_mov_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_mov_epi8(__U, __A); 
}

__m128i test_mm_mask_loadu_epi16(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_epi16
  // CHECK: @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mask_loadu_epi16(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi16(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_maskz_loadu_epi16(__U, __P); 
}

__m256i test_mm256_mask_loadu_epi16(__m256i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_epi16
  // CHECK: @llvm.masked.load.v16i16.p0v16i16(<16 x i16>* %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mask_loadu_epi16(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi16(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v16i16.p0v16i16(<16 x i16>* %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_maskz_loadu_epi16(__U, __P); 
}

__m128i test_mm_mask_loadu_epi8(__m128i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_mask_loadu_epi8(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi8(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: @test_mm_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_maskz_loadu_epi8(__U, __P); 
}

__m256i test_mm256_mask_loadu_epi8(__m256i __W, __mmask32 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v32i8.p0v32i8(<32 x i8>* %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_mask_loadu_epi8(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi8(__mmask32 __U, void const *__P) {
  // CHECK-LABEL: @test_mm256_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v32i8.p0v32i8(<32 x i8>* %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_maskz_loadu_epi8(__U, __P); 
}

void test_mm_mask_storeu_epi16(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v8i16.p0v8i16(<8 x i16> %{{.*}}, <8 x i16>* %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi16(void *__P, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v16i16.p0v16i16(<16 x i16> %{{.*}}, <16 x i16>* %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm256_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm_mask_storeu_epi8(void *__P, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %{{.*}}, <16 x i8>* %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm_mask_storeu_epi8(__P, __U, __A); 
}

void test_mm256_mask_storeu_epi8(void *__P, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v32i8.p0v32i8(<32 x i8> %{{.*}}, <32 x i8>* %{{.*}}, i32 1, <32 x i1> %{{.*}})
  return _mm256_mask_storeu_epi8(__P, __U, __A); 
}
__mmask16 test_mm_test_epi8_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_test_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return _mm_test_epi8_mask(__A, __B); 
}

__mmask16 test_mm_mask_test_epi8_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_test_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi8_mask(__U, __A, __B); 
}

__mmask32 test_mm256_test_epi8_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_test_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return _mm256_test_epi8_mask(__A, __B); 
}

__mmask32 test_mm256_mask_test_epi8_mask(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_test_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_test_epi8_mask(__U, __A, __B); 
}

__mmask8 test_mm_test_epi16_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_test_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  return _mm_test_epi16_mask(__A, __B); 
}

__mmask8 test_mm_mask_test_epi16_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_test_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm256_test_epi16_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_test_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  return _mm256_test_epi16_mask(__A, __B); 
}

__mmask16 test_mm256_mask_test_epi16_mask(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_test_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_test_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm_testn_epi8_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_testn_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return _mm_testn_epi8_mask(__A, __B); 
}

__mmask16 test_mm_mask_testn_epi8_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_testn_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_testn_epi8_mask(__U, __A, __B); 
}

__mmask32 test_mm256_testn_epi8_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_testn_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return _mm256_testn_epi8_mask(__A, __B); 
}

__mmask32 test_mm256_mask_testn_epi8_mask(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_testn_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_testn_epi8_mask(__U, __A, __B); 
}

__mmask8 test_mm_testn_epi16_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_testn_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return _mm_testn_epi16_mask(__A, __B); 
}

__mmask8 test_mm_mask_testn_epi16_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_testn_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_testn_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm256_testn_epi16_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_testn_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return _mm256_testn_epi16_mask(__A, __B); 
}

__mmask16 test_mm256_mask_testn_epi16_mask(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_testn_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_testn_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm_movepi8_mask(__m128i __A) {
  // CHECK-LABEL: @test_mm_movepi8_mask
  // CHECK: [[CMP:%.*]] = icmp slt <16 x i8> %{{.*}}, zeroinitializer
  // CHECK: bitcast <16 x i1> [[CMP]] to i16
  return _mm_movepi8_mask(__A); 
}

__mmask32 test_mm256_movepi8_mask(__m256i __A) {
  // CHECK-LABEL: @test_mm256_movepi8_mask
  // CHECK: [[CMP:%.*]] = icmp slt <32 x i8> %{{.*}}, zeroinitializer
  // CHECK: bitcast <32 x i1> [[CMP]] to i32
  return _mm256_movepi8_mask(__A); 
}

__m128i test_mm_movm_epi8(__mmask16 __A) {
  // CHECK-LABEL: @test_mm_movm_epi8
  // CHECK: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: %vpmovm2.i = sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_movm_epi8(__A); 
}

__m256i test_mm256_movm_epi8(__mmask32 __A) {
  // CHECK-LABEL: @test_mm256_movm_epi8
  // CHECK: %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: %vpmovm2.i = sext <32 x i1> %{{.*}} to <32 x i8>
  return _mm256_movm_epi8(__A); 
}

__m128i test_mm_movm_epi16(__mmask8 __A) {
  // CHECK-LABEL: @test_mm_movm_epi16
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %vpmovm2.i = sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_movm_epi16(__A); 
}

__m256i test_mm256_movm_epi16(__mmask16 __A) {
  // CHECK-LABEL: @test_mm256_movm_epi16
  // CHECK: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: %vpmovm2.i = sext <16 x i1> %{{.*}} to <16 x i16>
  return _mm256_movm_epi16(__A); 
}

__m128i test_mm_mask_broadcastb_epi8(__m128i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_broadcastb_epi8(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastb_epi8(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_broadcastb_epi8(__M, __A);
}

__m256i test_mm256_mask_broadcastb_epi8(__m256i __O, __mmask32 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_broadcastb_epi8(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastb_epi8(__mmask32 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_broadcastb_epi8(__M, __A);
}

__m128i test_mm_mask_broadcastw_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_broadcastw_epi16(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastw_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_broadcastw_epi16(__M, __A);
}

__m256i test_mm256_mask_broadcastw_epi16(__m256i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_broadcastw_epi16(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastw_epi16(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: @test_mm256_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_broadcastw_epi16(__M, __A);
}
__m128i test_mm_mask_set1_epi8 (__m128i __O, __mmask16 __M, char __A){
  // CHECK-LABEL: @test_mm_mask_set1_epi8
  // CHECK: insertelement <16 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_set1_epi8(__O, __M, __A);
}
__m128i test_mm_maskz_set1_epi8 ( __mmask16 __M, char __A){
  // CHECK-LABEL: @test_mm_maskz_set1_epi8
  // CHECK: insertelement <16 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_set1_epi8( __M, __A);
}

__m256i test_mm256_mask_set1_epi8(__m256i __O, __mmask32 __M, char __A) {
  // CHECK-LABEL: @test_mm256_mask_set1_epi8
  // CHECK: insertelement <32 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  // CHECK:  select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_set1_epi8(__O, __M, __A);
}

__m256i test_mm256_maskz_set1_epi8( __mmask32 __M, char __A) {
  // CHECK-LABEL: @test_mm256_maskz_set1_epi8
  // CHECK: insertelement <32 x i8> undef, i8 %{{.*}}, i32 0
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 1
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 2
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 3
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 4
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 5
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 6
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 7
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 8
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 9
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 10
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 11
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 12
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 13
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 14
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 15
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 16
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 17
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 18
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 19
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 20
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 21
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 22
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 23
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 24
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 25
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 26
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 27
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 28
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 29
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 30
  // CHECK: insertelement <32 x i8> %{{.*}}, i8 %{{.*}}, i32 31
  // CHECK:  select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_set1_epi8( __M, __A);
}


__m256i test_mm256_mask_set1_epi16(__m256i __O, __mmask16 __M, short __A) {
  // CHECK-LABEL: @test_mm256_mask_set1_epi16
  // CHECK: insertelement <16 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  // CHECK:  select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_set1_epi16(__O, __M, __A); 
}

__m256i test_mm256_maskz_set1_epi16(__mmask16 __M, short __A) {
  // CHECK-LABEL: @test_mm256_maskz_set1_epi16
  // CHECK: insertelement <16 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 8
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 9
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 10
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 11
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 12
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 13
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 14
  // CHECK: insertelement <16 x i16> %{{.*}}, i16 %{{.*}}, i32 15
  // CHECK:  select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_set1_epi16(__M, __A); 
}

__m128i test_mm_mask_set1_epi16(__m128i __O, __mmask8 __M, short __A) {
  // CHECK-LABEL: @test_mm_mask_set1_epi16
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_set1_epi16(__O, __M, __A); 
}

__m128i test_mm_maskz_set1_epi16(__mmask8 __M, short __A) {
  // CHECK-LABEL: @test_mm_maskz_set1_epi16
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 %{{.*}}, i32 7
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_set1_epi16(__M, __A); 
}
__m128i test_mm_permutexvar_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.mask.permvar.hi.128
  return _mm_permutexvar_epi16(__A, __B); 
}

__m128i test_mm_maskz_permutexvar_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.mask.permvar.hi.128
  return _mm_maskz_permutexvar_epi16(__M, __A, __B); 
}

__m128i test_mm_mask_permutexvar_epi16(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.mask.permvar.hi.128
  return _mm_mask_permutexvar_epi16(__W, __M, __A, __B); 
}

__m256i test_mm256_permutexvar_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.mask.permvar.hi.256
  return _mm256_permutexvar_epi16(__A, __B); 
}

__m256i test_mm256_maskz_permutexvar_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.mask.permvar.hi.256
  return _mm256_maskz_permutexvar_epi16(__M, __A, __B); 
}

__m256i test_mm256_mask_permutexvar_epi16(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.mask.permvar.hi.256
  return _mm256_mask_permutexvar_epi16(__W, __M, __A, __B); 
}
__m128i test_mm_mask_alignr_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}

__m128i test_mm_maskz_alignr_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_alignr_epi8(__U, __A, __B, 2); 
}

__m256i test_mm256_mask_alignr_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}

__m256i test_mm256_maskz_alignr_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_alignr_epi8(__U, __A, __B, 2); 
}

__m128i test_mm_dbsad_epu8(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_dbsad_epu8
  // CHECK: @llvm.x86.avx512.mask.dbpsadbw.128
  return _mm_dbsad_epu8(__A, __B, 170); 
}

__m128i test_mm_mask_dbsad_epu8(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_dbsad_epu8
  // CHECK: @llvm.x86.avx512.mask.dbpsadbw.128
  return _mm_mask_dbsad_epu8(__W, __U, __A, __B, 170); 
}

__m128i test_mm_maskz_dbsad_epu8(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_dbsad_epu8
  // CHECK: @llvm.x86.avx512.mask.dbpsadbw.128
  return _mm_maskz_dbsad_epu8(__U, __A, __B, 170); 
}

__m256i test_mm256_dbsad_epu8(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_dbsad_epu8
  // CHECK: @llvm.x86.avx512.mask.dbpsadbw.256
  return _mm256_dbsad_epu8(__A, __B, 170); 
}

__m256i test_mm256_mask_dbsad_epu8(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_dbsad_epu8
  // CHECK: @llvm.x86.avx512.mask.dbpsadbw.256
  return _mm256_mask_dbsad_epu8(__W, __U, __A, __B, 170); 
}

__m256i test_mm256_maskz_dbsad_epu8(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_dbsad_epu8
  // CHECK: @llvm.x86.avx512.mask.dbpsadbw.256
  return _mm256_maskz_dbsad_epu8(__U, __A, __B, 170); 
}
__mmask8 test_mm_movepi16_mask(__m128i __A) {
  // CHECK-LABEL: @test_mm_movepi16_mask
  // CHECK: [[CMP:%.*]] = icmp slt <8 x i16> %{{.*}}, zeroinitializer
  // CHECK: bitcast <8 x i1> [[CMP]] to i8
  return _mm_movepi16_mask(__A); 
}

__mmask16 test_mm256_movepi16_mask(__m256i __A) {
  // CHECK-LABEL: @test_mm256_movepi16_mask
  // CHECK: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer
  // CHECK: bitcast <16 x i1> [[CMP]] to i16
  return _mm256_movepi16_mask(__A); 
}

__m128i test_mm_mask_shufflehi_epi16(__m128i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_shufflehi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shufflehi_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_shufflehi_epi16(__mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_shufflehi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shufflehi_epi16(__U, __A, 5); 
}

__m128i test_mm_mask_shufflelo_epi16(__m128i __W, __mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_mask_shufflelo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shufflelo_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_maskz_shufflelo_epi16(__mmask32 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm_maskz_shufflelo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shufflelo_epi16(__U, __A, 5); 
}

__m256i test_mm256_mask_shufflehi_epi16(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_shufflehi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shufflehi_epi16(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_shufflehi_epi16(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_shufflehi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shufflehi_epi16(__U, __A, 5); 
}

__m256i test_mm256_mask_shufflelo_epi16(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_mask_shufflelo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shufflelo_epi16(__W, __U, __A, 5); 
}

__m256i test_mm256_maskz_shufflelo_epi16(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm256_maskz_shufflelo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shufflelo_epi16(__U, __A, 5); 
}

void test_mm_mask_cvtepi16_storeu_epi8 (void * __P, __mmask8 __M, __m128i __A)
{
 // CHECK-LABEL:@test_mm_mask_cvtepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmov.wb.mem.128
 _mm_mask_cvtepi16_storeu_epi8 (__P, __M, __A);
}

void test_mm_mask_cvtsepi16_storeu_epi8 (void * __P, __mmask8 __M, __m128i __A)
{
 // CHECK-LABEL:@test_mm_mask_cvtsepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovs.wb.mem.128
  _mm_mask_cvtsepi16_storeu_epi8 ( __P,  __M, __A);
}

void test_mm_mask_cvtusepi16_storeu_epi8 (void * __P, __mmask8 __M, __m128i __A)
{
 // CHECK-LABEL:@test_mm_mask_cvtusepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovus.wb.mem.128
  _mm_mask_cvtusepi16_storeu_epi8 (__P, __M, __A);
}

void test_mm256_mask_cvtusepi16_storeu_epi8 (void * __P, __mmask16 __M, __m256i __A)
{
 // CHECK-LABEL:@test_mm256_mask_cvtusepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovus.wb.mem.256
  _mm256_mask_cvtusepi16_storeu_epi8 ( __P, __M, __A);
}

void test_mm256_mask_cvtepi16_storeu_epi8 (void * __P, __mmask16 __M, __m256i __A)
{
 // CHECK-LABEL:@test_mm256_mask_cvtepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmov.wb.mem.256
  _mm256_mask_cvtepi16_storeu_epi8 ( __P,  __M, __A);
}

void test_mm256_mask_cvtsepi16_storeu_epi8 (void * __P, __mmask16 __M, __m256i __A)
{
 // CHECK-LABEL:@test_mm256_mask_cvtsepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovs.wb.mem.256
 _mm256_mask_cvtsepi16_storeu_epi8 ( __P, __M, __A);
}
