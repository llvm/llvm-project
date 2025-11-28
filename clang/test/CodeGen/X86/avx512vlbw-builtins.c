// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx10.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx10.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx10.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Wall -Werror -Wsign-conversion -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -target-feature +avx512vl -fno-signed-char -emit-llvm -o - -Wall -Werror -Wsign-conversion -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx10.1 -emit-llvm -o - -Wall -Werror -Wsign-conversion -fexperimental-new-constant-interpreter | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

__mmask32 test_mm256_cmpeq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpeq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpeq_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpeq_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpeq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpeq_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpeq_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpeq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpeq_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpeq_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpeq_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi8_mask
  // CHECK: icmp sgt <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpgt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpgt_epi8_mask
  // CHECK: icmp sgt <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpgt_epi8_mask
  // CHECK: icmp sgt <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpgt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpgt_epi8_mask
  // CHECK: icmp sgt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpgt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpgt_epi16_mask
  // CHECK: icmp sgt <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpgt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpgt_epi16_mask
  // CHECK: icmp sgt <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpgt_epi16_mask
  // CHECK: icmp sgt <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpgt_epi16_mask
  // CHECK: icmp sgt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpeq_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpeq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpeq_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpeq_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpeq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpeq_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpeq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpeq_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpeq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpeq_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpeq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpeq_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpeq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpeq_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpeq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpgt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpgt_epu8_mask
  // CHECK: icmp ugt <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpgt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpgt_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpgt_epu8_mask
  // CHECK: icmp ugt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpgt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpgt_epu16_mask
  // CHECK: icmp ugt <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpgt_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpgt_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpgt_epu16_mask
  // CHECK: icmp ugt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpgt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpgt_epu8_mask
  // CHECK: icmp ugt <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpgt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpgt_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpgt_epu8_mask
  // CHECK: icmp ugt <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpgt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpgt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpgt_epu16_mask
  // CHECK: icmp ugt <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpgt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpgt_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpgt_epu16_mask
  // CHECK: icmp ugt <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpgt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpge_epi8_mask
  // CHECK: icmp sge <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpge_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpge_epi8_mask
  // CHECK: icmp sge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpge_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpge_epu8_mask
  // CHECK: icmp uge <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpge_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpge_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpge_epu8_mask
  // CHECK: icmp uge <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpge_epi16_mask
  // CHECK: icmp sge <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpge_epi16_mask
  // CHECK: icmp sge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpge_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpge_epu16_mask
  // CHECK: icmp uge <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpge_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpge_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpge_epu16_mask
  // CHECK: icmp uge <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpge_epi8_mask
  // CHECK: icmp sge <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpge_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpge_epi8_mask
  // CHECK: icmp sge <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpge_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpge_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpge_epu8_mask
  // CHECK: icmp uge <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpge_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpge_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpge_epu8_mask
  // CHECK: icmp uge <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpge_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpge_epi16_mask
  // CHECK: icmp sge <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpge_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpge_epi16_mask
  // CHECK: icmp sge <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpge_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpge_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpge_epu16_mask
  // CHECK: icmp uge <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpge_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpge_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpge_epu16_mask
  // CHECK: icmp uge <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpge_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmple_epi8_mask
  // CHECK: icmp sle <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmple_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmple_epi8_mask
  // CHECK: icmp sle <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmple_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmple_epu8_mask
  // CHECK: icmp ule <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmple_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmple_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmple_epu8_mask
  // CHECK: icmp ule <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmple_epi16_mask
  // CHECK: icmp sle <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmple_epi16_mask
  // CHECK: icmp sle <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmple_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmple_epu16_mask
  // CHECK: icmp ule <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmple_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmple_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmple_epu16_mask
  // CHECK: icmp ule <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmple_epi8_mask
  // CHECK: icmp sle <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmple_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmple_epi8_mask
  // CHECK: icmp sle <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmple_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmple_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmple_epu8_mask
  // CHECK: icmp ule <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmple_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmple_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmple_epu8_mask
  // CHECK: icmp ule <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmple_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmple_epi16_mask
  // CHECK: icmp sle <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmple_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmple_epi16_mask
  // CHECK: icmp sle <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmple_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmple_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmple_epu16_mask
  // CHECK: icmp ule <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmple_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmple_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmple_epu16_mask
  // CHECK: icmp ule <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmple_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmplt_epi8_mask
  // CHECK: icmp slt <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmplt_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmplt_epi8_mask
  // CHECK: icmp slt <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmplt_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmplt_epu8_mask
  // CHECK: icmp ult <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmplt_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmplt_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmplt_epu8_mask
  // CHECK: icmp ult <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmplt_epi16_mask
  // CHECK: icmp slt <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmplt_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmplt_epi16_mask
  // CHECK: icmp slt <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmplt_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmplt_epu16_mask
  // CHECK: icmp ult <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmplt_epu16_mask(__a, __b);
}
TEST_CONSTEXPR(_mm_cmplt_epu16_mask(((__m128i)(__v8hu){12351, 47995, 11802, 16970, 16956, 13965, 33529, 18928}), ((__m128i)(__v8hu){48792, 59915, 50576, 62643, 3758, 16415, 7966, 39475})) == (__mmask8)0xAF);

__mmask8 test_mm_mask_cmplt_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmplt_epu16_mask
  // CHECK: icmp ult <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmplt_epi8_mask
  // CHECK: icmp slt <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmplt_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmplt_epi8_mask
  // CHECK: icmp slt <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmplt_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmplt_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmplt_epu8_mask
  // CHECK: icmp ult <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmplt_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmplt_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmplt_epu8_mask
  // CHECK: icmp ult <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmplt_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmplt_epi16_mask
  // CHECK: icmp slt <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmplt_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmplt_epi16_mask
  // CHECK: icmp slt <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmplt_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmplt_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmplt_epu16_mask
  // CHECK: icmp ult <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmplt_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmplt_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmplt_epu16_mask
  // CHECK: icmp ult <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmplt_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpneq_epi8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpneq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpneq_epi8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpneq_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpneq_epu8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmpneq_epu8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpneq_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpneq_epu8_mask
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpneq_epi16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpneq_epi16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpneq_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmpneq_epu16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmpneq_epu16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpneq_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmpneq_epu16_mask
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpneq_epi8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpneq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpneq_epi8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpneq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm256_cmpneq_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpneq_epu8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmpneq_epu8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpneq_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpneq_epu8_mask
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmpneq_epu8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpneq_epi16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpneq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpneq_epi16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpneq_epi16_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpneq_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmpneq_epu16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmpneq_epu16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpneq_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmpneq_epu16_mask
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmpneq_epu16_mask(__u, __a, __b);
}

__mmask16 test_mm_cmp_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmp_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmp_epi8_mask(__a, __b, 0);
}

// cmpeq tests
TEST_CONSTEXPR(_mm_cmpeq_epi8_mask(
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
) == (__mmask16)0xffff);

TEST_CONSTEXPR(_mm_cmpeq_epi8_mask(
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
) == (__mmask16)0xffff);

TEST_CONSTEXPR(_mm_cmpeq_epi8_mask(
    ((__m128i)(__v16qi){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}),
    ((__m128i)(__v16qi){-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128})
) == (__mmask16)0x0000);

TEST_CONSTEXPR(_mm_cmpeq_epi8_mask(
    ((__m128i)(__v16qi){-31, 90, -66, 3, 27, -22, -64, 111, -87, 105, -19, 0, 26, -111, 24, -72}),
    ((__m128i)(__v16qi){-84, -98, 20, -83, -98, 80, -46, -9, 22, -120, -123, 53, 117, -85, 50, 94})
) == (__mmask16)0x0000);

// cmpneq tests
TEST_CONSTEXPR(_mm_cmpneq_epi8_mask(
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
) == (__mmask16)0x0000);

TEST_CONSTEXPR(_mm_cmpneq_epi8_mask(
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
) == (__mmask16)0x0000);

TEST_CONSTEXPR(_mm_cmpneq_epi8_mask(
    ((__m128i)(__v16qi){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}),
    ((__m128i)(__v16qi){-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128})
) == (__mmask16)0xffff);

TEST_CONSTEXPR(_mm_cmpneq_epi8_mask(
    ((__m128i)(__v16qi){-2, 49, -5, -11, 21, -70, 121, -111, 20, 112, -121, 18, -34, -73, 89, 122}),
    ((__m128i)(__v16qi){14, 36, 2, 3, 118, 88, -126, -21, 104, -125, -1, 39, 99, -12, 35, -126})
) == (__mmask16)0xffff);

// cmplt tests
TEST_CONSTEXPR(_mm_cmplt_epi8_mask(
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
) == (__mmask16)0x0000);

TEST_CONSTEXPR(_mm_cmplt_epi8_mask(
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
) == (__mmask16)0x0000);

TEST_CONSTEXPR(_mm_cmplt_epi8_mask(
    ((__m128i)(__v16qi){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}),
    ((__m128i)(__v16qi){-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128})
) == (__mmask16)0x0000);

TEST_CONSTEXPR(_mm_cmplt_epi8_mask(
    ((__m128i)(__v16qi){-111, -10, -60, -123, -6, -110, -43, -32, -58, -7, 42, -128, -21, 24, 8, -101}),
    ((__m128i)(__v16qi){-108, 30, 71, 73, 20, 117, 63, -93, 79, -30, 99, -100, 34, 49, 83, 68})
) == (__mmask16)0xfd7f);

// cmple tests
TEST_CONSTEXPR(_mm_cmple_epi8_mask(
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmple_epi8_mask(
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmple_epi8_mask(
    ((__m128i)(__v16qi){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}),
    ((__m128i)(__v16qi){-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmple_epi8_mask(
    ((__m128i)(__v16qi){122, 6, -22, -94, 78, -35, -43, -16, -69, 124, -2, 24, -117, 8, -17, 118}),
    ((__m128i)(__v16qi){53, -50, 104, 11, 63, -77, -25, 102, 46, 62, 27, -28, -61, 68, 40, -65})
) == (__mmask16)0x75cc);

// cmpge tests
TEST_CONSTEXPR(_mm_cmpge_epi8_mask(
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpge_epi8_mask(
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpge_epi8_mask(
    ((__m128i)(__v16qi){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}),
    ((__m128i)(__v16qi){-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpge_epi8_mask(
    ((__m128i)(__v16qi){-11, 0, 97, 123, -48, 105, 26, -118, 62, -86, -94, -32, 14, -4, -50, 72}),
    ((__m128i)(__v16qi){-33, 49, 22, 31, -4, -81, 6, -22, 40, 127, -1, -106, 6, -64, 12, 8})
) == (__mmask16)0xb96d);

// cmpgt tests
TEST_CONSTEXPR(_mm_cmpgt_epi8_mask(
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
    ((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmpgt_epi8_mask(
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
    ((__m128i)(__v16qi){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmpgt_epi8_mask(
    ((__m128i)(__v16qi){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}),
    ((__m128i)(__v16qi){-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpgt_epi8_mask(
    ((__m128i)(__v16qi){-127, 37, -123, -60, 98, -68, -115, 96, 80, -27, -105, 64, -6, -51, -45, -81}),
    ((__m128i)(__v16qi){-124, 59, 8, 59, 122, 47, -74, 120, 19, -43, -33, -76, 7, -22, -24, -14})
) == (__mmask16)0x0b00);

// _mm_cmp*_epi16_mask tests
// cmpeq tests
TEST_CONSTEXPR(_mm_cmpeq_epi16_mask(
    ((__m128i)(__v8hi){-3442, 30871, -12144, -1454, -22826, -2857, -15573, 31071}),
    ((__m128i)(__v8hi){18051, 1305, 19877, 25257, -1276, -27260, -2189, -12561})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epi16_mask(
    ((__m128i)(__v8hi){-30000, -12116, -26732, -25017, 1825, 22572, -23178, -19747}),
    ((__m128i)(__v8hi){2100, 15453, 17949, -29756, 17479, -18870, -22571, -17226})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epi16_mask(
    ((__m128i)(__v8hi){31014, -1995, -10083, -9515, 10226, -21525, 253, 30999}),
    ((__m128i)(__v8hi){-25332, -30218, 31439, 15484, -31344, 17581, -30883, -1427})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epi16_mask(
    ((__m128i)(__v8hi){-9653, -20924, -20387, 285, -14945, 19791, 21735, -31655}),
    ((__m128i)(__v8hi){-19129, -16192, 6453, -3246, -22032, 9464, -15122, -12084})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epi16_mask(
    ((__m128i)(__v8hi){-24008, 21674, 8299, -31133, 23486, -29407, -15611, -7613}),
    ((__m128i)(__v8hi){10079, -14333, 2802, 21711, -10849, 11468, -528, -18259})
) == (__mmask8)0x00);

// cmpneq tests
TEST_CONSTEXPR(_mm_cmpneq_epi16_mask(
    ((__m128i)(__v8hi){29745, -23358, -12495, -13908, 3162, -15181, -20831, -9264}),
    ((__m128i)(__v8hi){-32594, -25364, -11888, -32704, 25399, -14592, -8293, -19662})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epi16_mask(
    ((__m128i)(__v8hi){21379, -15275, -31427, -29628, 14433, 30414, -24752, 119}),
    ((__m128i)(__v8hi){-23650, -17663, 2882, -8677, -13639, -6226, -17564, -17024})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epi16_mask(
    ((__m128i)(__v8hi){5659, -8091, 28040, 7676, 3226, -12548, 29893, 12766}),
    ((__m128i)(__v8hi){23731, -3987, -27073, 5995, -5765, 10343, 21504, -18513})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epi16_mask(
    ((__m128i)(__v8hi){17386, 29612, 28099, -8524, -7042, -23864, 13718, 16802}),
    ((__m128i)(__v8hi){8658, 5451, 782, -9208, 8027, 14983, -4011, -31827})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epi16_mask(
    ((__m128i)(__v8hi){-24378, 20934, 25612, 4427, -29446, 26311, -2533, -27963}),
    ((__m128i)(__v8hi){28538, -27232, 21741, -31546, -18083, 17112, -25750, -28016})
) == (__mmask8)0xff);

// cmplt tests
TEST_CONSTEXPR(_mm_cmplt_epi16_mask(
    ((__m128i)(__v8hi){13828, -2305, -29543, 13467, 13750, 4765, -20421, -1340}),
    ((__m128i)(__v8hi){17531, 9112, -7503, -2262, 12476, -24405, 543, 24076})
) == (__mmask8)0xc7);
TEST_CONSTEXPR(_mm_cmplt_epi16_mask(
    ((__m128i)(__v8hi){26544, 10625, -23568, -9115, -27479, -15046, -2294, 17617}),
    ((__m128i)(__v8hi){-17598, -27694, 3374, 32022, 25371, -3401, -1466, 18741})
) == (__mmask8)0xfc);
TEST_CONSTEXPR(_mm_cmplt_epi16_mask(
    ((__m128i)(__v8hi){-20731, -29738, 15746, 14938, -18069, -3978, 25561, -23366}),
    ((__m128i)(__v8hi){12163, -26845, -19607, 12781, -2854, 23771, 6504, 21136})
) == (__mmask8)0xb3);
TEST_CONSTEXPR(_mm_cmplt_epi16_mask(
    ((__m128i)(__v8hi){2923, -3968, 11644, 28855, -6225, 32503, -10027, -18296}),
    ((__m128i)(__v8hi){17481, -24809, 24058, -5767, -1033, 16133, -8332, -24426})
) == (__mmask8)0x55);
TEST_CONSTEXPR(_mm_cmplt_epi16_mask(
    ((__m128i)(__v8hi){-23454, -12570, 28483, -2443, -10879, 21881, -27324, 10746}),
    ((__m128i)(__v8hi){31572, -12990, 27653, -25336, 26491, 21811, 19562, -27562})
) == (__mmask8)0x51);

// cmple tests
TEST_CONSTEXPR(_mm_cmple_epi16_mask(
    ((__m128i)(__v8hi){-13404, 5773, -2912, 12257, -21386, 12244, -25523, 28508}),
    ((__m128i)(__v8hi){2730, -15074, 7642, -15989, -17979, -8818, -28069, 1577})
) == (__mmask8)0x15);
TEST_CONSTEXPR(_mm_cmple_epi16_mask(
    ((__m128i)(__v8hi){-30681, 29588, -21231, 17008, 4229, -12947, -6153, -2183}),
    ((__m128i)(__v8hi){-9893, 12451, 3221, 2418, 6992, -9738, 21000, 7839})
) == (__mmask8)0xf5);
TEST_CONSTEXPR(_mm_cmple_epi16_mask(
    ((__m128i)(__v8hi){9571, 3844, 19158, 8637, 27094, -16447, -31005, 11095}),
    ((__m128i)(__v8hi){3929, 901, 31907, 15912, 5595, -6562, -9664, -13180})
) == (__mmask8)0x6c);
TEST_CONSTEXPR(_mm_cmple_epi16_mask(
    ((__m128i)(__v8hi){-29653, -19891, 32155, -13473, -1745, 23019, -30659, 9319}),
    ((__m128i)(__v8hi){11054, 23285, -27270, 31021, -2491, 7728, 30175, 3553})
) == (__mmask8)0x4b);
TEST_CONSTEXPR(_mm_cmple_epi16_mask(
    ((__m128i)(__v8hi){10008, -28465, -29830, 22527, 8820, -32356, 15584, 23957}),
    ((__m128i)(__v8hi){-4691, 13730, -28788, -17664, 14597, -29281, -30463, 7606})
) == (__mmask8)0x36);

// cmpge tests
TEST_CONSTEXPR(_mm_cmpge_epi16_mask(
    ((__m128i)(__v8hi){-11140, 18900, -28192, 12396, 12691, -17637, -5639, 13640}),
    ((__m128i)(__v8hi){13120, 27235, 31917, 5559, 31947, -16460, 12054, -11750})
) == (__mmask8)0x88);
TEST_CONSTEXPR(_mm_cmpge_epi16_mask(
    ((__m128i)(__v8hi){29089, -626, 23207, 7124, 19598, 15564, 6422, 9504}),
    ((__m128i)(__v8hi){-5950, -29409, -25182, 30585, -31424, -12209, 1669, -31761})
) == (__mmask8)0xf7);
TEST_CONSTEXPR(_mm_cmpge_epi16_mask(
    ((__m128i)(__v8hi){-20490, -26633, -14323, -14815, -23375, 18639, 30382, 10679}),
    ((__m128i)(__v8hi){-19313, 26898, -1867, -31542, 25972, -24148, 14949, -21235})
) == (__mmask8)0xe8);
TEST_CONSTEXPR(_mm_cmpge_epi16_mask(
    ((__m128i)(__v8hi){-1967, 20611, 17473, -3220, -8421, 23008, -4566, 32760}),
    ((__m128i)(__v8hi){6049, 26996, 3995, 3649, 22926, -31207, 5407, -25394})
) == (__mmask8)0xa4);
TEST_CONSTEXPR(_mm_cmpge_epi16_mask(
    ((__m128i)(__v8hi){-29193, 9029, 15883, -20070, 12934, -20531, 32059, -16251}),
    ((__m128i)(__v8hi){-392, -23600, 21384, 3664, -23762, -25166, 5219, -5042})
) == (__mmask8)0x72);

// cmpgt tests
TEST_CONSTEXPR(_mm_cmpgt_epi16_mask(
    ((__m128i)(__v8hi){15267, -11951, -5449, 23797, -2214, -9874, 9646, -21342}),
    ((__m128i)(__v8hi){9306, 8824, 29545, 26005, 31680, -21215, 32078, 24049})
) == (__mmask8)0x21);
TEST_CONSTEXPR(_mm_cmpgt_epi16_mask(
    ((__m128i)(__v8hi){31831, 31959, -26932, -29861, 24449, -4255, 28223, -21248}),
    ((__m128i)(__v8hi){-7535, -25898, 21764, 20515, 18797, -7540, -9244, -8230})
) == (__mmask8)0x73);
TEST_CONSTEXPR(_mm_cmpgt_epi16_mask(
    ((__m128i)(__v8hi){7503, -3478, 26774, -3300, 21598, -20865, 6044, -8590}),
    ((__m128i)(__v8hi){5765, 29485, 26086, 13029, 7790, -24192, -14148, -25392})
) == (__mmask8)0xf5);
TEST_CONSTEXPR(_mm_cmpgt_epi16_mask(
    ((__m128i)(__v8hi){5627, -22902, 25994, -19341, -30428, 6898, 29163, -9331}),
    ((__m128i)(__v8hi){31016, -1554, 23610, -25442, 28669, 22077, 28653, 22826})
) == (__mmask8)0x4c);
TEST_CONSTEXPR(_mm_cmpgt_epi16_mask(
    ((__m128i)(__v8hi){-16014, 12488, 2614, -25164, -8107, -20887, -21726, 32065}),
    ((__m128i)(__v8hi){-32191, 26305, -6600, 3970, -31254, 30169, 22872, -2017})
) == (__mmask8)0x95);

// Test cases for _mm_cmp*_epu8_mask
// cmpeq tests
TEST_CONSTEXPR(_mm_cmpeq_epu8_mask(
    ((__m128i)(__v16qu){16, 136, 12, 57, 51, 16, 77, 210, 75, 252, 75, 9, 225, 11, 166, 94}),
    ((__m128i)(__v16qu){225, 224, 24, 209, 45, 121, 99, 61, 10, 97, 27, 158, 101, 233, 34, 12})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmpeq_epu8_mask(
    ((__m128i)(__v16qu){224, 224, 206, 52, 121, 157, 252, 238, 48, 100, 124, 229, 100, 234, 30, 208}),
    ((__m128i)(__v16qu){61, 95, 167, 40, 44, 237, 43, 102, 132, 252, 109, 244, 185, 115, 205, 2})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmpeq_epu8_mask(
    ((__m128i)(__v16qu){12, 206, 18, 41, 83, 198, 246, 178, 153, 221, 101, 107, 46, 116, 173, 28}),
    ((__m128i)(__v16qu){127, 214, 9, 253, 109, 138, 48, 204, 110, 34, 91, 206, 157, 216, 22, 157})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmpeq_epu8_mask(
    ((__m128i)(__v16qu){239, 95, 183, 64, 22, 28, 232, 140, 235, 134, 213, 29, 47, 9, 75, 224}),
    ((__m128i)(__v16qu){224, 78, 249, 228, 241, 3, 175, 176, 226, 242, 76, 169, 231, 49, 135, 97})
) == (__mmask16)0x0000);
TEST_CONSTEXPR(_mm_cmpeq_epu8_mask(
    ((__m128i)(__v16qu){145, 106, 8, 210, 56, 39, 72, 146, 99, 151, 112, 16, 160, 80, 140, 70}),
    ((__m128i)(__v16qu){203, 124, 25, 113, 140, 85, 126, 152, 170, 25, 6, 121, 146, 40, 113, 57})
) == (__mmask16)0x0000);

// cmpneq tests
TEST_CONSTEXPR(_mm_cmpneq_epu8_mask(
    ((__m128i)(__v16qu){225, 50, 34, 45, 215, 179, 100, 173, 136, 135, 252, 121, 217, 29, 82, 98}),
    ((__m128i)(__v16qu){160, 184, 95, 94, 53, 251, 207, 101, 37, 176, 104, 54, 21, 228, 221, 180})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpneq_epu8_mask(
    ((__m128i)(__v16qu){25, 167, 250, 22, 212, 147, 54, 115, 226, 5, 240, 116, 158, 154, 120, 134}),
    ((__m128i)(__v16qu){142, 156, 34, 216, 124, 89, 209, 226, 119, 133, 230, 19, 204, 218, 38, 124})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpneq_epu8_mask(
    ((__m128i)(__v16qu){19, 219, 73, 161, 64, 71, 158, 100, 163, 157, 73, 111, 208, 90, 168, 136}),
    ((__m128i)(__v16qu){4, 15, 146, 12, 104, 227, 125, 133, 28, 148, 58, 111, 107, 172, 181, 33})
) == (__mmask16)0xf7ff);
TEST_CONSTEXPR(_mm_cmpneq_epu8_mask(
    ((__m128i)(__v16qu){134, 53, 253, 117, 213, 129, 192, 67, 243, 215, 233, 223, 121, 100, 180, 173}),
    ((__m128i)(__v16qu){4, 198, 27, 51, 149, 246, 28, 192, 18, 157, 214, 77, 119, 147, 1, 28})
) == (__mmask16)0xffff);
TEST_CONSTEXPR(_mm_cmpneq_epu8_mask(
    ((__m128i)(__v16qu){241, 10, 186, 52, 173, 193, 93, 240, 187, 30, 147, 130, 221, 70, 210, 44}),
    ((__m128i)(__v16qu){65, 145, 226, 92, 171, 211, 64, 61, 82, 183, 135, 205, 124, 25, 81, 244})
) == (__mmask16)0xffff);

// cmplt tests
TEST_CONSTEXPR(_mm_cmplt_epu8_mask(
    ((__m128i)(__v16qu){188, 3, 27, 241, 77, 48, 233, 136, 166, 7, 248, 212, 73, 41, 134, 236}),
    ((__m128i)(__v16qu){99, 67, 183, 99, 117, 11, 225, 198, 61, 141, 198, 145, 244, 68, 198, 52})
) == (__mmask16)0x7296);
TEST_CONSTEXPR(_mm_cmplt_epu8_mask(
    ((__m128i)(__v16qu){90, 197, 4, 61, 147, 4, 63, 30, 93, 225, 249, 120, 130, 225, 204, 27}),
    ((__m128i)(__v16qu){131, 238, 224, 204, 79, 190, 139, 127, 145, 229, 163, 118, 119, 15, 122, 71})
) == (__mmask16)0x83ef);
TEST_CONSTEXPR(_mm_cmplt_epu8_mask(
    ((__m128i)(__v16qu){75, 44, 98, 1, 152, 251, 197, 192, 27, 36, 33, 183, 80, 24, 217, 121}),
    ((__m128i)(__v16qu){214, 10, 95, 115, 65, 179, 194, 230, 93, 92, 209, 78, 32, 88, 119, 223})
) == (__mmask16)0xa789);
TEST_CONSTEXPR(_mm_cmplt_epu8_mask(
    ((__m128i)(__v16qu){143, 136, 99, 215, 124, 156, 108, 234, 94, 186, 245, 21, 253, 27, 218, 93}),
    ((__m128i)(__v16qu){54, 248, 105, 21, 188, 253, 148, 114, 168, 96, 151, 167, 185, 152, 37, 1})
) == (__mmask16)0x2976);
TEST_CONSTEXPR(_mm_cmplt_epu8_mask(
    ((__m128i)(__v16qu){155, 8, 76, 39, 55, 79, 74, 78, 129, 144, 169, 84, 32, 112, 201, 226}),
    ((__m128i)(__v16qu){185, 107, 11, 239, 21, 120, 63, 105, 153, 148, 182, 2, 0, 181, 144, 100})
) == (__mmask16)0x27ab);

// cmple tests
TEST_CONSTEXPR(_mm_cmple_epu8_mask(
    ((__m128i)(__v16qu){89, 172, 39, 168, 219, 32, 80, 82, 190, 122, 23, 37, 15, 28, 217, 18}),
    ((__m128i)(__v16qu){122, 245, 45, 82, 41, 70, 1, 117, 26, 169, 89, 160, 147, 78, 105, 226})
) == (__mmask16)0xbea7);
TEST_CONSTEXPR(_mm_cmple_epu8_mask(
    ((__m128i)(__v16qu){175, 250, 230, 32, 79, 107, 111, 206, 244, 181, 147, 110, 95, 178, 7, 32}),
    ((__m128i)(__v16qu){173, 117, 248, 184, 244, 190, 240, 65, 119, 215, 67, 30, 252, 224, 86, 42})
) == (__mmask16)0xf27c);
TEST_CONSTEXPR(_mm_cmple_epu8_mask(
    ((__m128i)(__v16qu){14, 90, 183, 71, 139, 11, 177, 242, 104, 66, 206, 15, 214, 178, 91, 120}),
    ((__m128i)(__v16qu){255, 3, 220, 86, 47, 19, 208, 93, 76, 231, 175, 247, 111, 146, 51, 9})
) == (__mmask16)0x0a6d);
TEST_CONSTEXPR(_mm_cmple_epu8_mask(
    ((__m128i)(__v16qu){104, 98, 30, 11, 120, 187, 239, 183, 105, 114, 48, 201, 5, 5, 88, 10}),
    ((__m128i)(__v16qu){145, 250, 12, 166, 92, 215, 45, 237, 122, 132, 41, 25, 93, 15, 88, 32})
) == (__mmask16)0xf3ab);
TEST_CONSTEXPR(_mm_cmple_epu8_mask(
    ((__m128i)(__v16qu){31, 238, 238, 163, 38, 109, 134, 50, 251, 250, 68, 2, 132, 238, 236, 148}),
    ((__m128i)(__v16qu){26, 223, 228, 186, 240, 53, 148, 56, 106, 196, 76, 246, 114, 102, 237, 127})
) == (__mmask16)0x4cd8);

// cmpge tests
TEST_CONSTEXPR(_mm_cmpge_epu8_mask(
    ((__m128i)(__v16qu){209, 5, 40, 54, 128, 0, 44, 248, 51, 117, 117, 106, 57, 231, 172, 234}),
    ((__m128i)(__v16qu){127, 165, 98, 192, 238, 10, 85, 220, 132, 177, 167, 233, 173, 232, 215, 90})
) == (__mmask16)0x8081);
TEST_CONSTEXPR(_mm_cmpge_epu8_mask(
    ((__m128i)(__v16qu){41, 233, 30, 65, 230, 138, 250, 66, 252, 130, 119, 231, 237, 228, 122, 34}),
    ((__m128i)(__v16qu){71, 153, 86, 224, 213, 87, 212, 43, 234, 194, 148, 66, 243, 238, 59, 221})
) == (__mmask16)0x49f2);
TEST_CONSTEXPR(_mm_cmpge_epu8_mask(
    ((__m128i)(__v16qu){198, 89, 26, 129, 103, 127, 255, 110, 186, 70, 43, 10, 12, 216, 254, 209}),
    ((__m128i)(__v16qu){83, 120, 28, 162, 196, 225, 222, 24, 172, 30, 154, 144, 65, 56, 94, 146})
) == (__mmask16)0xe3c1);
TEST_CONSTEXPR(_mm_cmpge_epu8_mask(
    ((__m128i)(__v16qu){59, 193, 244, 59, 108, 159, 53, 244, 201, 6, 253, 224, 211, 74, 146, 0}),
    ((__m128i)(__v16qu){107, 244, 12, 240, 201, 25, 88, 46, 179, 174, 124, 99, 112, 4, 69, 16})
) == (__mmask16)0x7da4);
TEST_CONSTEXPR(_mm_cmpge_epu8_mask(
    ((__m128i)(__v16qu){165, 179, 185, 74, 129, 89, 42, 170, 195, 35, 151, 20, 240, 155, 245, 254}),
    ((__m128i)(__v16qu){104, 175, 83, 169, 96, 144, 164, 61, 6, 46, 150, 36, 177, 188, 77, 147})
) == (__mmask16)0xd597);

// cmpgt tests
TEST_CONSTEXPR(_mm_cmpgt_epu8_mask(
    ((__m128i)(__v16qu){38, 190, 130, 195, 187, 214, 113, 247, 178, 29, 139, 40, 157, 185, 136, 248}),
    ((__m128i)(__v16qu){142, 87, 246, 176, 127, 177, 41, 209, 155, 139, 105, 140, 191, 70, 15, 224})
) == (__mmask16)0xe5fa);
TEST_CONSTEXPR(_mm_cmpgt_epu8_mask(
    ((__m128i)(__v16qu){64, 243, 108, 190, 6, 147, 219, 249, 213, 244, 43, 185, 134, 167, 89, 98}),
    ((__m128i)(__v16qu){249, 226, 180, 166, 53, 137, 244, 172, 105, 31, 30, 43, 218, 96, 36, 140})
) == (__mmask16)0x6faa);
TEST_CONSTEXPR(_mm_cmpgt_epu8_mask(
    ((__m128i)(__v16qu){66, 24, 106, 97, 243, 223, 111, 54, 33, 138, 171, 254, 228, 155, 195, 178}),
    ((__m128i)(__v16qu){204, 154, 118, 221, 206, 135, 119, 52, 84, 105, 228, 68, 184, 244, 85, 243})
) == (__mmask16)0x5ab0);
TEST_CONSTEXPR(_mm_cmpgt_epu8_mask(
    ((__m128i)(__v16qu){222, 143, 193, 146, 136, 215, 84, 82, 172, 158, 189, 199, 246, 114, 149, 116}),
    ((__m128i)(__v16qu){152, 16, 193, 198, 99, 235, 130, 181, 211, 129, 39, 210, 94, 169, 4, 11})
) == (__mmask16)0xd613);
TEST_CONSTEXPR(_mm_cmpgt_epu8_mask(
    ((__m128i)(__v16qu){25, 67, 207, 46, 203, 0, 89, 144, 93, 235, 192, 245, 29, 67, 227, 66}),
    ((__m128i)(__v16qu){59, 110, 28, 206, 112, 88, 82, 187, 108, 82, 224, 47, 184, 168, 201, 105})
) == (__mmask16)0x4a54);

// Test cases for _mm_cmp*_epu16_mask
// cmpeq tests
TEST_CONSTEXPR(_mm_cmpeq_epu16_mask(
    ((__m128i)(__v8hu){26017, 41021, 1946, 38661, 30561, 19297, 15060, 15737}),
    ((__m128i)(__v8hu){8185, 18330, 37519, 42141, 61247, 31228, 6264, 59456})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epu16_mask(
    ((__m128i)(__v8hu){36575, 60206, 3432, 10195, 34831, 43860, 40303, 49709}),
    ((__m128i)(__v8hu){1432, 64285, 48336, 3804, 18528, 49977, 15597, 51759})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epu16_mask(
    ((__m128i)(__v8hu){25178, 38322, 33003, 43178, 39760, 44915, 5994, 55537}),
    ((__m128i)(__v8hu){34277, 7150, 30624, 59335, 51255, 12845, 65164, 3858})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epu16_mask(
    ((__m128i)(__v8hu){57800, 61974, 13596, 50836, 40573, 216, 31321, 38806}),
    ((__m128i)(__v8hu){28357, 61850, 47986, 47656, 1664, 36978, 15274, 60611})
) == (__mmask8)0x00);
TEST_CONSTEXPR(_mm_cmpeq_epu16_mask(
    ((__m128i)(__v8hu){57064, 51949, 42152, 43038, 65515, 7797, 23751, 12389}),
    ((__m128i)(__v8hu){38499, 21192, 9584, 15701, 4826, 6826, 47257, 20923})
) == (__mmask8)0x00);

// cmpneq tests
TEST_CONSTEXPR(_mm_cmpneq_epu16_mask(
    ((__m128i)(__v8hu){14257, 58047, 39173, 29624, 18365, 12002, 39567, 46161}),
    ((__m128i)(__v8hu){39492, 1657, 38848, 30475, 33784, 22129, 16165, 39862})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epu16_mask(
    ((__m128i)(__v8hu){35289, 33907, 13787, 15259, 39250, 33252, 47224, 23905}),
    ((__m128i)(__v8hu){62303, 41983, 19416, 4638, 48011, 59449, 46768, 20704})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epu16_mask(
    ((__m128i)(__v8hu){43035, 24266, 48861, 29042, 24252, 18719, 2345, 15378}),
    ((__m128i)(__v8hu){12179, 44184, 13531, 23032, 7839, 49161, 39424, 17456})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epu16_mask(
    ((__m128i)(__v8hu){30106, 39497, 21733, 28765, 52826, 54726, 50106, 61970}),
    ((__m128i)(__v8hu){34749, 64882, 7430, 10309, 26658, 22489, 34795, 35513})
) == (__mmask8)0xff);
TEST_CONSTEXPR(_mm_cmpneq_epu16_mask(
    ((__m128i)(__v8hu){17033, 42616, 6158, 53144, 14513, 29528, 57905, 62537}),
    ((__m128i)(__v8hu){38447, 51378, 320, 16853, 48397, 49715, 53668, 43839})
) == (__mmask8)0xff);

// cmplt tests
TEST_CONSTEXPR(_mm_cmplt_epu16_mask(
    ((__m128i)(__v8hu){12351, 47995, 11802, 16970, 16956, 13965, 33529, 18928}),
    ((__m128i)(__v8hu){48792, 59915, 50576, 62643, 3758, 16415, 7966, 39475})
) == (__mmask8)0xaf);
TEST_CONSTEXPR(_mm_cmplt_epu16_mask(
    ((__m128i)(__v8hu){64376, 821, 6646, 56557, 41027, 54650, 38293, 65034}),
    ((__m128i)(__v8hu){16194, 13150, 9765, 40643, 62592, 32105, 37358, 43042})
) == (__mmask8)0x16);
TEST_CONSTEXPR(_mm_cmplt_epu16_mask(
    ((__m128i)(__v8hu){5197, 38111, 57400, 21773, 6778, 55251, 43022, 23599}),
    ((__m128i)(__v8hu){25453, 44508, 11034, 51861, 28507, 3586, 50043, 22852})
) == (__mmask8)0x5b);
TEST_CONSTEXPR(_mm_cmplt_epu16_mask(
    ((__m128i)(__v8hu){45352, 2829, 63457, 8858, 31243, 17792, 61465, 39468}),
    ((__m128i)(__v8hu){36906, 65156, 12702, 36312, 25463, 30740, 17784, 16985})
) == (__mmask8)0x2a);
TEST_CONSTEXPR(_mm_cmplt_epu16_mask(
    ((__m128i)(__v8hu){55773, 8388, 56635, 32082, 18204, 53351, 33529, 17025}),
    ((__m128i)(__v8hu){40397, 47032, 24886, 11409, 5864, 42244, 24984, 9087})
) == (__mmask8)0x02);

// cmple tests
TEST_CONSTEXPR(_mm_cmple_epu16_mask(
    ((__m128i)(__v8hu){7117, 51838, 34713, 59818, 13657, 57172, 51354, 44657}),
    ((__m128i)(__v8hu){43684, 29399, 39679, 60107, 61794, 25979, 20734, 59740})
) == (__mmask8)0x9d);
TEST_CONSTEXPR(_mm_cmple_epu16_mask(
    ((__m128i)(__v8hu){34191, 12870, 38578, 50725, 13944, 40695, 8891, 56035}),
    ((__m128i)(__v8hu){15935, 55419, 10154, 43268, 20049, 46161, 14264, 35221})
) == (__mmask8)0x72);
TEST_CONSTEXPR(_mm_cmple_epu16_mask(
    ((__m128i)(__v8hu){33955, 56125, 59463, 56265, 62541, 27516, 15841, 10852}),
    ((__m128i)(__v8hu){110, 25696, 52299, 36663, 26130, 22670, 40665, 36384})
) == (__mmask8)0xc0);
TEST_CONSTEXPR(_mm_cmple_epu16_mask(
    ((__m128i)(__v8hu){30109, 45669, 64670, 2417, 24233, 12069, 21306, 31838}),
    ((__m128i)(__v8hu){41978, 40368, 4025, 37501, 5634, 43889, 27907, 243})
) == (__mmask8)0x69);
TEST_CONSTEXPR(_mm_cmple_epu16_mask(
    ((__m128i)(__v8hu){30467, 14437, 9940, 54360, 37732, 61408, 1077, 33377}),
    ((__m128i)(__v8hu){21329, 57575, 30740, 42725, 26374, 53724, 12, 11808})
) == (__mmask8)0x06);

// cmpge tests
TEST_CONSTEXPR(_mm_cmpge_epu16_mask(
    ((__m128i)(__v8hu){18151, 20516, 44819, 57353, 65147, 3123, 59838, 46999}),
    ((__m128i)(__v8hu){55970, 31815, 16503, 37882, 46380, 47912, 2076, 33014})
) == (__mmask8)0xdc);
TEST_CONSTEXPR(_mm_cmpge_epu16_mask(
    ((__m128i)(__v8hu){60160, 48803, 35124, 42510, 62783, 18767, 26220, 16209}),
    ((__m128i)(__v8hu){22162, 54330, 56847, 11199, 46429, 7012, 12165, 42137})
) == (__mmask8)0x79);
TEST_CONSTEXPR(_mm_cmpge_epu16_mask(
    ((__m128i)(__v8hu){15118, 50748, 64103, 47033, 11223, 61427, 3201, 61926}),
    ((__m128i)(__v8hu){19094, 30041, 5839, 62591, 59904, 1530, 40604, 34353})
) == (__mmask8)0xa6);
TEST_CONSTEXPR(_mm_cmpge_epu16_mask(
    ((__m128i)(__v8hu){37865, 12467, 45289, 60251, 60268, 3908, 51781, 35681}),
    ((__m128i)(__v8hu){30915, 32324, 1515, 24432, 15318, 33790, 22826, 64258})
) == (__mmask8)0x5d);
TEST_CONSTEXPR(_mm_cmpge_epu16_mask(
    ((__m128i)(__v8hu){53950, 52988, 13868, 25190, 31823, 62039, 43379, 51291}),
    ((__m128i)(__v8hu){35593, 54830, 49773, 10890, 44742, 48266, 45280, 14226})
) == (__mmask8)0xa9);

// cmpgt tests
TEST_CONSTEXPR(_mm_cmpgt_epu16_mask(
    ((__m128i)(__v8hu){53569, 10332, 63903, 42861, 45572, 36510, 50522, 36590}),
    ((__m128i)(__v8hu){20661, 10192, 56056, 56939, 27398, 16004, 27165, 43131})
) == (__mmask8)0x77);
TEST_CONSTEXPR(_mm_cmpgt_epu16_mask(
    ((__m128i)(__v8hu){64844, 511, 53950, 40143, 20601, 53105, 34756, 60447}),
    ((__m128i)(__v8hu){20846, 55132, 57724, 35384, 25368, 41097, 3764, 52879})
) == (__mmask8)0xe9);
TEST_CONSTEXPR(_mm_cmpgt_epu16_mask(
    ((__m128i)(__v8hu){64715, 31454, 34266, 43085, 52260, 7304, 25587, 43911}),
    ((__m128i)(__v8hu){25076, 3569, 11981, 11259, 63537, 32468, 23699, 9601})
) == (__mmask8)0xcf);
TEST_CONSTEXPR(_mm_cmpgt_epu16_mask(
    ((__m128i)(__v8hu){57954, 26888, 1708, 125, 19459, 44354, 29760, 23713}),
    ((__m128i)(__v8hu){2188, 13828, 28514, 39528, 42286, 41311, 28270, 11063})
) == (__mmask8)0xe3);
TEST_CONSTEXPR(_mm_cmpgt_epu16_mask(
    ((__m128i)(__v8hu){36435, 57842, 17322, 33066, 51263, 58618, 57550, 23212}),
    ((__m128i)(__v8hu){150, 39532, 62935, 34670, 57126, 9790, 21078, 3593})
) == (__mmask8)0xe3);



// tests for _mm256_cmp*_mask

// cmpeq
TEST_CONSTEXPR(_mm256_cmpeq_epi8_mask(
    ((__m256i)(__v32qi){65, 123, -1, -81, 85, -32, -1, -128, -69, 118, -114, -22, 97, 48, -64, -53, -76, -112, 40, 123, 35, -90, 125, 1, 4, 117, -13, -78, -118, -26, -59, -27}),
    ((__m256i)(__v32qi){-122, -37, 93, 112, -113, 36, 72, -110, -75, 57, 52, 69, -51, -85, 36, -62, 45, -90, -45, 102, -87, 114, 34, -97, 13, 15, -125, -72, 74, 115, -92, -5})
) == (__mmask32)0x0);

TEST_CONSTEXPR(_mm256_cmpeq_epi8_mask(
    ((__m256i)(__v32qi){9, 23, 36, -105, -32, 41, -58, 107, -118, 13, -53, 49, -71, 127, 81, -50, -27, 102, -120, 63, -26, 64, -11, 34, 95, -124, -42, 98, 121, -39, 54, 86}),
    ((__m256i)(__v32qi){126, 115, -9, -58, -80, 105, -98, -57, -43, -16, 3, 27, -43, -56, 11, -81, 15, -64, -30, -55, -6, 19, 116, 115, -41, -23, 110, 36, 114, 4, 90, 62})
) == (__mmask32)0x0);

TEST_CONSTEXPR(_mm256_cmpeq_epi8_mask(
    ((__m256i)(__v32qi){-107, 108, 114, 56, 115, 41, -5, 77, -124, 123, -3, -117, 101, 89, 41, 87, -43, 95, 62, 45, 115, 30, -125, 81, 46, -28, -34, 125, -99, -110, 106, 34}),
    ((__m256i)(__v32qi){36, 53, 102, -84, 11, 21, 55, 87, 28, -36, -1, 7, 7, 112, 33, 100, 69, -96, 12, 44, 14, -38, -23, -99, -79, 1, -77, 82, 3, 32, 5, -123})
) == (__mmask32)0x0);

TEST_CONSTEXPR(_mm256_cmpeq_epi8_mask(
    ((__m256i)(__v32qi){-117, 57, -55, -65, 41, 101, 41, 91, 55, -54, -33, 55, 14, -29, 117, 124, 72, -98, -57, 77, -126, 9, -86, 45, -128, 86, -57, -12, 37, -81, -108, 107}),
    ((__m256i)(__v32qi){-94, -25, 126, -53, -72, 84, -88, -56, -20, -117, 97, 127, 63, 53, -3, -88, -83, -34, 60, -82, 42, -86, 126, -20, 16, -33, 97, -90, -95, 48, 88, -124})
) == (__mmask32)0x0);

TEST_CONSTEXPR(_mm256_cmpeq_epi8_mask(
    ((__m256i)(__v32qi){-62, 71, 111, 70, 20, 26, 34, -11, 49, 122, 80, 96, -30, 103, -11, 73, -105, -20, -98, 12, 106, 7, 61, 25, 39, -6, -109, 2, -26, -86, -77, -94}),
    ((__m256i)(__v32qi){-2, 120, 99, -40, -72, -83, 73, 80, -74, -110, 4, 91, -33, -24, 38, -122, -41, 115, 63, 43, 113, -43, 121, 46, -30, 92, 52, -117, -75, -113, -31, -47})
) == (__mmask32)0x0);

// cmpneq
TEST_CONSTEXPR(_mm256_cmpneq_epi8_mask(
    ((__m256i)(__v32qi){108, -96, -35, 20, 25, -22, -28, 44, 4, -14, 99, -112, -94, 110, -2, -88, 70, 99, -80, -31, -9, -10, 81, 16, -125, -91, -112, 79, -117, 123, 14, 3}),
    ((__m256i)(__v32qi){80, 69, 32, -43, -124, -8, 64, 27, -1, 119, -62, 5, -42, 5, 39, -44, 21, 46, -48, 60, 117, 51, -96, 53, 35, 29, -53, -99, 80, 89, 41, 33})
) == (__mmask32)0xffffffff);

TEST_CONSTEXPR(_mm256_cmpneq_epi8_mask(
    ((__m256i)(__v32qi){-86, 105, -105, 99, -125, -40, -75, -82, 27, 72, 68, -23, -95, -128, -87, -103, 86, 123, -96, -124, 64, 109, -74, 64, 17, 28, 27, 22, 108, 36, -16, -1}),
    ((__m256i)(__v32qi){-43, -26, 56, -30, -66, 18, 52, 90, 60, 5, -110, 0, -36, 83, 94, -29, -36, -5, -39, 106, -36, 120, -14, 23, 38, 48, -5, -95, -65, -118, -27, -104})
) == (__mmask32)0xffffffff);

TEST_CONSTEXPR(_mm256_cmpneq_epi8_mask(
    ((__m256i)(__v32qi){-111, 117, -70, -12, -61, -20, -78, 75, 115, -70, 25, -51, -76, 44, 93, -73, 47, -23, 115, -40, -93, -29, 76, 66, 32, 19, 113, -71, 61, -50, 19, 89}),
    ((__m256i)(__v32qi){17, -32, -44, -72, 56, 46, 71, -77, 117, 122, -15, 27, 94, -61, -8, -41, 31, -10, 65, -96, 61, 41, -114, -59, -38, 86, -18, -79, -4, 30, -121, -35})
) == (__mmask32)0xffffffff);

TEST_CONSTEXPR(_mm256_cmpneq_epi8_mask(
    ((__m256i)(__v32qi){-93, -108, 39, 110, 62, -79, 114, 47, -82, -38, -27, -114, 66, 66, 98, 12, 53, -37, 13, -45, -120, -46, -108, 28, -79, 19, -24, 44, -127, -115, 123, 113}),
    ((__m256i)(__v32qi){4, -109, 82, 27, 53, -127, -113, -102, -51, -59, -122, -117, 6, -11, -16, -72, 107, 78, 10, -6, -34, -47, 104, 33, -69, 53, 33, 83, -24, 52, 109, 124})
) == (__mmask32)0xffffffff);

TEST_CONSTEXPR(_mm256_cmpneq_epi8_mask(
    ((__m256i)(__v32qi){1, -32, 61, 98, 123, 34, 99, -77, -34, 68, 101, 37, 29, -99, 33, -22, 85, 13, 26, 85, 8, -17, -124, 19, 69, -40, -32, 58, -35, -4, -125, -112}),
    ((__m256i)(__v32qi){30, 45, 85, -71, -111, 111, -82, 107, -71, -4, -46, 8, 52, -107, 107, -14, 33, 35, 31, 116, -75, -33, -82, -60, 26, 40, 82, 85, -78, 97, 104, -91})
) == (__mmask32)0xffffffff);

// cmplt 
//
TEST_CONSTEXPR(_mm256_cmplt_epi8_mask(
    ((__m256i)(__v32qi){101, -24, 65, 23, -24, -46, 114, 45, -95, 125, -113, 27, -121, -41, -9, -87, -41, 40, -105, -5, 10, 68, 18, -121, -73, -89, -22, 95, 85, 112, 88, -46}),
    ((__m256i)(__v32qi){80, -31, 9, -9, -61, -18, 97, -115, -102, -16, 7, -106, -73, 119, 43, -45, 22, 60, -110, -83, 100, 97, 102, 30, 22, -10, 106, 116, -90, 75, 34, 41})
) == (__mmask32)0x8ff3f420);

TEST_CONSTEXPR(_mm256_cmplt_epi8_mask(
    ((__m256i)(__v32qi){-26, 127, 107, 82, 115, 115, 0, -61, -46, -98, -128, 114, -44, 43, -121, 60, 43, 99, 77, 67, -24, 100, 22, 121, 44, 42, -102, 30, 64, 25, -83, -56}),
    ((__m256i)(__v32qi){118, -39, -70, -97, 76, 95, 100, 5, 122, -79, 5, 114, -18, 27, 110, -58, -62, -30, -72, -19, 37, 55, 42, 36, 93, -117, 97, -40, 13, 119, 83, -120})
) == (__mmask32)0x655057c1);

TEST_CONSTEXPR(_mm256_cmplt_epi8_mask(
    ((__m256i)(__v32qi){33, -60, -34, -23, 5, -90, -96, 11, 24, 110, -17, 104, 74, -111, 66, -53, 84, 28, -124, 10, 104, 114, -32, -13, 124, 76, -101, -18, -46, -41, 127, -11}),
    ((__m256i)(__v32qi){-26, 94, -39, -19, -122, -107, -6, 122, 43, 6, 19, -121, 103, 26, 48, -9, 58, 120, 35, 29, -60, 11, -24, 104, -1, -44, -22, -112, -95, 93, -13, 34})
) == (__mmask32)0xa4ceb5ca);

TEST_CONSTEXPR(_mm256_cmplt_epi8_mask(
    ((__m256i)(__v32qi){-97, -105, -72, 99, 9, 46, -15, -69, -79, 71, 87, 93, 58, -50, 112, -119, 90, 25, -18, 83, 3, -37, -56, -23, 119, -49, -52, 12, -47, -84, -41, -58}),
    ((__m256i)(__v32qi){107, -19, -95, -7, 63, -25, -55, 4, 109, 51, 90, -62, 53, -9, -69, -58, -98, -35, 32, -25, 40, -53, -66, 86, -121, -92, 116, 93, 122, -111, 105, -114})
) == (__mmask32)0x5c94a593);

TEST_CONSTEXPR(_mm256_cmplt_epi8_mask(
    ((__m256i)(__v32qi){72, -110, -26, 40, -96, -73, 13, 72, 112, -34, -25, 49, -82, 84, 21, -78, 67, 55, -2, -23, 2, 8, -103, 60, -30, 107, 87, -17, -6, -86, 118, -115}),
    ((__m256i)(__v32qi){64, 40, -74, 122, 111, -128, 21, 62, 22, 99, -10, -128, -118, 70, -61, 117, 0, 20, -28, 77, 45, 88, -7, -74, -34, 85, -15, 3, -16, -4, 127, 46})
) == (__mmask32)0xe878865a);

// cmple
TEST_CONSTEXPR(_mm256_cmple_epi8_mask(
    ((__m256i)(__v32qi){-5, -85, 40, -29, -25, -124, 8, 119, 118, -3, -106, 115, -89, -35, -31, 66, 67, -95, 35, 86, 44, 12, -65, -18, -118, 57, 76, 41, -48, 123, -107, -28}),
    ((__m256i)(__v32qi){-49, 7, -66, -90, -71, 44, 67, -65, 100, -83, 93, 95, -20, 91, 73, -16, -127, -45, -89, 86, -3, -112, -6, 87, 34, -76, -16, -89, 87, -77, -29, 18})
) == (__mmask32)0xd1ca7462);

TEST_CONSTEXPR(_mm256_cmple_epi8_mask(
    ((__m256i)(__v32qi){-37, 68, 22, 32, -46, 5, -1, -117, -54, -127, 67, 54, -8, -98, -3, 13, -120, 119, 13, -70, -38, -85, -97, -6, -49, -89, 95, 83, 76, 5, -125, 92}),
    ((__m256i)(__v32qi){-98, 43, -13, 119, -58, 95, 64, 35, -41, 105, -114, -16, -52, -18, 126, -20, 22, -87, 90, -127, 33, -42, 45, 103, -117, 41, -65, -76, 34, -14, -69, -9})
) == (__mmask32)0x42f563e8);

TEST_CONSTEXPR(_mm256_cmple_epi8_mask(
    ((__m256i)(__v32qi){-53, 43, -115, -57, -116, 70, -116, -123, 66, -12, -32, 7, -30, -1, -25, 22, 14, -46, -87, -26, 110, -69, -83, 16, 20, 77, 43, -128, 68, -15, 104, 6}),
    ((__m256i)(__v32qi){54, -84, 126, -69, 102, 80, 97, 19, 21, 39, -26, -12, -26, -114, -3, 10, 0, 89, 89, -73, -119, -13, 6, 97, -44, 64, -49, 92, 25, 55, 16, -39})
) == (__mmask32)0x28e656f5);

TEST_CONSTEXPR(_mm256_cmple_epi8_mask(
    ((__m256i)(__v32qi){-23, -75, -97, -103, -121, 60, -80, 61, 118, 51, 89, 118, -71, 111, -28, -92, 12, 94, 40, 10, 26, -38, 14, -82, 42, 18, -104, 120, 43, 126, 84, -51}),
    ((__m256i)(__v32qi){-52, -76, 27, -52, -21, -105, -128, 101, -18, 60, -70, -24, -72, -19, 47, 108, 7, 22, -37, -50, -51, 58, 0, -87, 12, 126, -57, 81, 73, 111, -71, -81})
) == (__mmask32)0x1620c29c);

TEST_CONSTEXPR(_mm256_cmple_epi8_mask(
    ((__m256i)(__v32qi){91, -39, -23, 58, 12, 71, -56, 34, -100, 111, -77, -48, -101, -25, -126, -5, 100, 47, 110, 6, 73, 52, -92, 34, 96, -98, 41, -60, 88, 5, -101, -11}),
    ((__m256i)(__v32qi){73, -81, 68, -86, -8, -16, -47, 56, -30, 18, 3, -29, -73, -98, -37, 57, -30, 79, 26, -71, -99, 15, 61, 18, -12, 59, -114, 14, -105, 38, -63, 101})
) == (__mmask32)0xea42ddc4);


//  cmpgt
TEST_CONSTEXPR(_mm256_cmpgt_epi8_mask(
    ((__m256i)(__v32qi){0, -46, -12, -123, 80, 117, -10, -29, -68, 98, -70, 53, -99, -120, -92, 70, 91, -45, -55, -38, -73, 15, 20, -109, 57, 72, 65, -101, -38, 17, -59, 12}),
    ((__m256i)(__v32qi){52, 104, -97, 48, 90, 98, 1, 97, -28, -98, 106, 104, -64, -90, -9, 0, -28, 69, -54, -20, -29, -77, 3, 99, 48, 56, 21, -38, 61, 125, 30, -30})
) == (__mmask32)0x87618224);

TEST_CONSTEXPR(_mm256_cmpgt_epi8_mask(
    ((__m256i)(__v32qi){-96, 2, 2, -27, -8, 85, -108, 39, -115, 31, -122, 36, -75, -57, 25, 109, 121, -1, -85, -24, 91, -61, -17, 83, -111, 82, -81, -112, 104, 61, -24, 14}),
    ((__m256i)(__v32qi){-104, 33, -53, -37, 96, 32, 14, -41, -10, 108, 49, 102, 91, -56, 22, -102, -67, -105, -102, 32, 42, -97, 38, -67, -23, -119, -100, 82, -12, -78, -61, 62})
) == (__mmask32)0x76b7c0ad);

TEST_CONSTEXPR(_mm256_cmpgt_epi8_mask(
    ((__m256i)(__v32qi){-125, 53, -122, -91, 112, 37, 49, -112, 27, -79, -72, -79, -22, 87, 87, -124, 127, -87, -64, 126, 59, 9, 35, -17, -81, -76, -91, -33, 22, -18, -122, 13}),
    ((__m256i)(__v32qi){-45, -16, -111, 98, -102, -46, 26, 124, 41, 8, -45, -118, -97, -66, -44, 108, 110, -3, 83, 84, -11, 98, 25, 86, 77, 84, -100, 3, 72, 90, 102, -2})
) == (__mmask32)0x84597872);

TEST_CONSTEXPR(_mm256_cmpgt_epi8_mask(
    ((__m256i)(__v32qi){-126, -6, 92, -46, -126, -20, -126, -124, -121, 39, 118, -106, -48, -118, -83, 36, -125, 28, -48, -91, -31, 68, -38, 123, -69, -35, -10, 18, -82, 48, 114, -121}),
    ((__m256i)(__v32qi){42, -56, 92, -106, -50, -20, 53, -36, -40, -77, 76, 119, -104, 76, -52, -65, 34, -126, 69, 108, 33, 108, 3, 93, -60, -113, -104, -8, -6, 55, 105, 52})
) == (__mmask32)0x4e82960a);

TEST_CONSTEXPR(_mm256_cmpgt_epi8_mask(
    ((__m256i)(__v32qi){17, -85, 47, -4, 36, 38, -60, 33, -35, 89, -87, 23, 104, 30, -49, 41, 1, 112, -7, 22, 126, 26, -57, -23, 67, -124, -120, 62, 72, 126, 43, -49}),
    ((__m256i)(__v32qi){-48, -107, -125, 53, 78, 121, 92, 17, -51, 79, 116, -21, -52, 26, 63, -71, -86, -99, -11, 13, 75, -120, -113, -87, 115, -94, 74, -82, 117, -6, -47, 103})
) == (__mmask32)0x68ffbb87);
//  cmpge
TEST_CONSTEXPR(_mm256_cmpge_epi8_mask(
    ((__m256i)(__v32qi){55, -102, 124, 13, 94, 13, -23, 98, -101, -124, 116, 90, -88, -123, 72, 28, -29, 10, -48, 11, -102, 7, 81, -88, -82, -9, -116, -31, -74, 45, 0, 27}),
    ((__m256i)(__v32qi){-125, -35, -45, -16, -125, 93, 81, -70, 43, -93, 116, 46, -82, -61, -54, -55, -18, 62, 11, -4, 106, -2, -26, -27, -114, -70, 33, -27, 37, -116, 77, 94})
) == (__mmask32)0x2368cc9d);

TEST_CONSTEXPR(_mm256_cmpge_epi8_mask(
    ((__m256i)(__v32qi){34, -31, 101, 49, -69, 57, 94, -127, -32, 22, 10, -79, 29, 68, -31, 71, 96, 99, 17, -39, -41, -121, 103, -22, -74, -112, -110, -121, -97, -86, -122, -88}),
    ((__m256i)(__v32qi){24, 119, -121, 29, -13, -19, -48, 50, -109, 48, 38, 54, 85, 87, 48, 5, -3, 4, 21, -110, -119, 15, 114, -113, 95, -100, 105, 78, -77, 43, 30, 79})
) == (__mmask32)0x9b816d);

TEST_CONSTEXPR(_mm256_cmpge_epi8_mask(
    ((__m256i)(__v32qi){81, 85, -34, -6, 112, -36, -99, 18, -47, 3, -54, 56, -12, -86, -33, 60, 92, 49, 82, 0, 103, -109, 99, -25, -62, -15, 64, -119, -8, 13, -77, -56}),
    ((__m256i)(__v32qi){90, -42, -23, -101, 113, -66, 87, 66, 72, -103, 67, -56, -128, 118, 112, 58, 56, 83, -7, -103, -91, 40, -64, 91, -108, -62, 22, 41, -32, 78, 92, -2})
) == (__mmask32)0x175d9a2a);

TEST_CONSTEXPR(_mm256_cmpge_epi8_mask(
    ((__m256i)(__v32qi){-80, 84, -7, -109, 70, -116, -120, -87, -75, 92, -42, -12, -78, -81, 1, -117, -93, -24, -19, -16, -110, -120, -71, 124, 75, 20, -69, 65, -67, -39, -50, -56}),
    ((__m256i)(__v32qi){75, -102, -40, -112, 36, -112, 34, 50, 4, -54, -60, -35, -44, 107, -94, -110, 47, -116, -101, 9, -44, 42, -121, 93, 26, -13, -127, 37, -17, 112, 60, -10})
) == (__mmask32)0xfc64e1e);

TEST_CONSTEXPR(_mm256_cmpge_epi8_mask(
    ((__m256i)(__v32qi){66, 9, 79, -49, 14, 49, 81, -4, 33, -113, 18, -85, -77, -112, -119, 19, 80, -74, 105, 120, 76, -123, -12, 8, 98, 72, -37, -29, -120, 17, -107, 101}),
    ((__m256i)(__v32qi){119, 10, 36, 76, -116, 12, -9, 75, 53, -52, -9, -57, -28, -80, -27, 34, 106, 44, -109, 20, -10, 119, 116, 16, 36, 9, -14, -61, 6, -70, 26, 84})
) == (__mmask32)0xab1c0474);

__mmask16 test_mm_mask_cmp_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmp_epi8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmp_epi8_mask(__u, __a, __b, 0);
}

__mmask16 test_mm_cmp_epu8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmp_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_cmp_epu8_mask(__a, __b, 0);
}

__mmask16 test_mm_mask_cmp_epu8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmp_epu8_mask
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm_mask_cmp_epu8_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmp_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epi16_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmp_epi16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epi16_mask(__u, __a, __b, 0);
}

__mmask8 test_mm_cmp_epu16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_cmp_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_cmp_epu16_mask(__a, __b, 0);
}

__mmask8 test_mm_mask_cmp_epu16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: test_mm_mask_cmp_epu16_mask
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return (__mmask8)_mm_mask_cmp_epu16_mask(__u, __a, __b, 0);
}

__mmask32 test_mm256_cmp_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmp_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmp_epi8_mask(__a, __b, 0);
}

__mmask32 test_mm256_mask_cmp_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmp_epi8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmp_epi8_mask(__u, __a, __b, 0);
}

__mmask32 test_mm256_cmp_epu8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmp_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_cmp_epu8_mask(__a, __b, 0);
}

__mmask32 test_mm256_mask_cmp_epu8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmp_epu8_mask
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return (__mmask32)_mm256_mask_cmp_epu8_mask(__u, __a, __b, 0);
}

__mmask16 test_mm256_cmp_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmp_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmp_epi16_mask(__a, __b, 0);
}

__mmask16 test_mm256_mask_cmp_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmp_epi16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmp_epi16_mask(__u, __a, __b, 0);
}

__mmask16 test_mm256_cmp_epu16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_cmp_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_cmp_epu16_mask(__a, __b, 0);
}

__mmask16 test_mm256_mask_cmp_epu16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: test_mm256_mask_cmp_epu16_mask
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return (__mmask16)_mm256_mask_cmp_epu16_mask(__u, __a, __b, 0);
}


__m256i test_mm256_mask_add_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B){
  //CHECK-LABEL: test_mm256_mask_add_epi8
  //CHECK: add <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_add_epi8(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v32qi(_mm256_mask_add_epi8((__m256i)(__v32qs){ 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0x20676BFF, (__m256i)(__v32qs){ -64, -65, 66, 67, 68, -69, -70, -71, -72, 73, 74, -75, 76, 77, -78, 79, -80, 81, 82, -83, 84, -85, 86, 87, -88, -89, -90, 91, -92, 93, 94, 95}, (__m256i)(__v32qs){ -1, -2, 3, -4, 5, -6, 7, 8, 9, -10, 11, -12, 13, 14, 15, 16, -17, 18, 19, 20, -21, 22, -23, -24, 25, 26, -27, -28, -29, 30, -31, 32}), -65, -67, 69, 63, 73, -75, -63, -63, -63, 63, 99, -87, 99, 91, -63, 99, -97, 99, 101, 99, 99, -63, 63, 99, 99, 99, 99, 99, 99, 123, 99, 99));

__m256i test_mm256_maskz_add_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_maskz_add_epi8
  //CHECK: add <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_add_epi8(__U , __A, __B);
}

TEST_CONSTEXPR(match_v32qi(_mm256_maskz_add_epi8(0x20676BFF, (__m256i)(__v32qs){ -64, -65, 66, 67, 68, -69, -70, -71, -72, 73, 74, -75, 76, 77, -78, 79, -80, 81, 82, -83, 84, -85, 86, 87, -88, -89, -90, 91, -92, 93, 94, 95}, (__m256i)(__v32qs){ -1, -2, 3, -4, 5, -6, 7, 8, 9, -10, 11, -12, 13, 14, 15, 16, -17, 18, 19, 20, -21, 22, -23, -24, 25, 26, -27, -28, -29, 30, -31, 32}),  -65, -67, 69, 63, 73, -75, -63, -63, -63, 63, 0, -87, 0, 91, -63, 0, -97, 99, 101, 0, 0, -63, 63, 0, 0, 0, 0, 0, 0, 123, 0, 0));

__m256i test_mm256_mask_add_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_mask_add_epi16
  //CHECK: add <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_add_epi16(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_add_epi16((__m256i)(__v16hi){ 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0x5A21, (__m256i)(__v16hi){ -32, -33, 34, -35, -36, 37, -38, -39, -40, -41, -42, 43, -44, -45, 46, 47}, (__m256i)(__v16hi){ -1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, -12, -13, 14, -15, -16}), -33, 99, 99, 99, 99, 31, 99, 99, 99, -31, 99, 31, -57, 99, 31, 99));

__m256i test_mm256_maskz_add_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_maskz_add_epi16
  //CHECK: add <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_add_epi16(__U , __A, __B);
}

TEST_CONSTEXPR(match_v16hi(_mm256_maskz_add_epi16(0x5A21, (__m256i)(__v16hi){ -32, -33, 34, -35, -36, 37, -38, -39, -40, -41, -42, 43, -44, -45, 46, 47}, (__m256i)(__v16hi){ -1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, -12, -13, 14, -15, -16}),  -33, 0, 0, 0, 0, 31, 0, 0, 0, -31, 0, 31, -57, 0, 31, 0));

__m256i test_mm256_mask_sub_epi8 (__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_mask_sub_epi8
  //CHECK: sub <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_sub_epi8(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v32qi(_mm256_mask_sub_epi8((__m256i)(__v32qs){ 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0x20676BFF, (__m256i)(__v32qs){ -64, -65, 66, 67, 68, -69, -70, -71, -72, 73, 74, -75, 76, 77, -78, 79, -80, 81, 82, -83, 84, -85, 86, 87, -88, -89, -90, 91, -92, 93, 94, 95}, (__m256i)(__v32qs){ -1, -2, 3, -4, 5, -6, 7, 8, 9, -10, 11, -12, 13, 14, 15, 16, -17, 18, 19, 20, -21, 22, -23, -24, 25, 26, -27, -28, -29, 30, -31, 32}), -63, -63, 63, 71, 63, -63, -77, -79, -81, 83, 99, -63, 99, 63, -93, 99, -63, 63, 63, 99, 99, -107, 109, 99, 99, 99, 99, 99, 99, 63, 99, 99));

__m256i test_mm256_maskz_sub_epi8 (__mmask32 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_maskz_sub_epi8
  //CHECK: sub <32 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_sub_epi8(__U , __A, __B);
}

TEST_CONSTEXPR(match_v32qi(_mm256_maskz_sub_epi8(0x20676BFF, (__m256i)(__v32qs){ -64, -65, 66, 67, 68, -69, -70, -71, -72, 73, 74, -75, 76, 77, -78, 79, -80, 81, 82, -83, 84, -85, 86, 87, -88, -89, -90, 91, -92, 93, 94, 95}, (__m256i)(__v32qs){ -1, -2, 3, -4, 5, -6, 7, 8, 9, -10, 11, -12, 13, 14, 15, 16, -17, 18, 19, 20, -21, 22, -23, -24, 25, 26, -27, -28, -29, 30, -31, 32}),  -63, -63, 63, 71, 63, -63, -77, -79, -81, 83, 0, -63, 0, 63, -93, 0, -63, 63, 63, 0, 0, -107, 109, 0, 0, 0, 0, 0, 0, 63, 0, 0));

__m256i test_mm256_mask_sub_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_mask_sub_epi16
  //CHECK: sub <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sub_epi16(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_sub_epi16((__m256i)(__v16hi){ 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0x5A21, (__m256i)(__v16hi){ -32, -33, 34, -35, -36, 37, -38, -39, -40, -41, -42, 43, -44, -45, 46, 47}, (__m256i)(__v16hi){ -1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, -12, -13, 14, -15, -16}), -31, 99, 99, 99, 99, 43, 99, 99, 99, -51, 99, 55, -31, 99, 61, 99));

__m256i test_mm256_maskz_sub_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_maskz_sub_epi16
  //CHECK: sub <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sub_epi16(__U , __A, __B);
}

TEST_CONSTEXPR(match_v16hi(_mm256_maskz_sub_epi16(0x5A21, (__m256i)(__v16hi){ -32, -33, 34, -35, -36, 37, -38, -39, -40, -41, -42, 43, -44, -45, 46, 47}, (__m256i)(__v16hi){ -1, 2, 3, -4, 5, -6, 7, 8, 9, 10, -11, -12, -13, 14, -15, -16}),  -31, 0, 0, 0, 0, 43, 0, 0, 0, -51, 0, 55, -31, 0, 61, 0));

__m128i test_mm_mask_add_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_mask_add_epi8
  //CHECK: add <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_add_epi8(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v16qi(_mm_mask_add_epi8((__m128i)(__v16qs){ 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0x5693, (__m128i)(__v16qs){ 32, 33, 34, -35, -36, -37, 38, 39, -40, 41, -42, 43, -44, -45, 46, -47}, (__m128i)(__v16qs){ 1, 2, -3, -4, 5, -6, -7, 8, 9, -10, 11, 12, 13, 14, -15, 16}), 33, 35, 99, 99, -31, 99, 99, 47, 99, 31, -31, 99, -31, 99, 31, 99));

__m128i test_mm_maskz_add_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_maskz_add_epi8
  //CHECK: add <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_add_epi8(__U , __A, __B);
}

TEST_CONSTEXPR(match_v16qi(_mm_maskz_add_epi8(0x5693, (__m128i)(__v16qs){ 32, 33, 34, -35, -36, -37, 38, 39, -40, 41, -42, 43, -44, -45, 46, -47}, (__m128i)(__v16qs){ 1, 2, -3, -4, 5, -6, -7, 8, 9, -10, 11, 12, 13, 14, -15, 16}),  33, 35, 0, 0, -31, 0, 0, 47, 0, 31, -31, 0, -31, 0, 31, 0));

__m128i test_mm_mask_add_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_mask_add_epi16
  //CHECK: add <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_add_epi16(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_add_epi16((__m128i)(__v8hi){ 99, 99, 99, 99, 99, 99, 99, 99}, 0x8D, (__m128i)(__v8hi){ -16, -17, -18, -19, -20, 21, 22, 23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, -6, 7, -8}), -15, 99, -15, -23, 99, 99, 99, 15));

__m128i test_mm_maskz_add_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_maskz_add_epi16
  //CHECK: add <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_add_epi16(__U , __A, __B);
}

TEST_CONSTEXPR(match_v8hi(_mm_maskz_add_epi16(0x8D, (__m128i)(__v8hi){ -16, -17, -18, -19, -20, 21, 22, 23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, -6, 7, -8}),  -15, 0, -15, -23, 0, 0, 0, 15));

__m128i test_mm_mask_sub_epi8 (__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_mask_sub_epi8
  //CHECK: sub <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_sub_epi8(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v16qi(_mm_mask_sub_epi8((__m128i)(__v16qs){ 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0x5693, (__m128i)(__v16qs){ 32, 33, 34, -35, -36, -37, 38, 39, -40, 41, -42, 43, -44, -45, 46, -47}, (__m128i)(__v16qs){ 1, 2, -3, -4, 5, -6, -7, 8, 9, -10, 11, 12, 13, 14, -15, 16}), 31, 31, 99, 99, -41, 99, 99, 31, 99, 51, -53, 99, -57, 99, 61, 99));

__m128i test_mm_maskz_sub_epi8 (__mmask16 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_maskz_sub_epi8
  //CHECK: sub <16 x i8> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_sub_epi8(__U , __A, __B);
}

TEST_CONSTEXPR(match_v16qi(_mm_maskz_sub_epi8(0x5693, (__m128i)(__v16qs){ 32, 33, 34, -35, -36, -37, 38, 39, -40, 41, -42, 43, -44, -45, 46, -47}, (__m128i)(__v16qs){ 1, 2, -3, -4, 5, -6, -7, 8, 9, -10, 11, 12, 13, 14, -15, 16}),  31, 31, 0, 0, -41, 0, 0, 31, 0, 51, -53, 0, -57, 0, 61, 0));

__m128i test_mm_mask_sub_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_mask_sub_epi16
  //CHECK: sub <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sub_epi16(__W, __U , __A, __B);
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_sub_epi16((__m128i)(__v8hi){ 99, 99, 99, 99, 99, 99, 99, 99}, 0x8D, (__m128i)(__v8hi){ -16, -17, -18, -19, -20, 21, 22, 23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, -6, 7, -8}), -17, 99, -21, -15, 99, 99, 99, 31));

__m128i test_mm_maskz_sub_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_maskz_sub_epi16
  //CHECK: sub <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sub_epi16(__U , __A, __B);
}

TEST_CONSTEXPR(match_v8hi(_mm_maskz_sub_epi16(0x8D, (__m128i)(__v8hi){ -16, -17, -18, -19, -20, 21, 22, 23}, (__m128i)(__v8hi){ 1, 2, 3, -4, -5, -6, 7, -8}),  -17, 0, -21, -15, 0, 0, 0, 31));

__m256i test_mm256_mask_mullo_epi16 (__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_mask_mullo_epi16
  //CHECK: mul <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mullo_epi16(__W, __U , __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_mullo_epi16((__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, 0x00FF, (__m256i)(__v16hi){+2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17}, (__m256i)(__v16hi){-3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18}), -6, -12, -20, -30, -42, -56, -72, -90, -9, +10, -11, +12, -13, +14, -15, +16));

__m256i test_mm256_maskz_mullo_epi16 (__mmask16 __U, __m256i __A, __m256i __B) {
  //CHECK-LABEL: test_mm256_maskz_mullo_epi16
  //CHECK: mul <16 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mullo_epi16(__U , __A, __B);
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_mullo_epi16(0x00FF, (__m256i)(__v16hi){+2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17}, (__m256i)(__v16hi){-3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18}), -6, -12, -20, -30, -42, -56, -72, -90, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_mullo_epi16 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_mask_mullo_epi16
  //CHECK: mul <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mullo_epi16(__W, __U , __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_mullo_epi16((__m128i)(__v8hi){-1, +2, -3, +4, -5, +6, -7, +8}, 0x0F, (__m128i)(__v8hi){+2, -3, +4, -5, +6, -7, +8, -9}, (__m128i)(__v8hi){-3, +4, -5, +6, -7, +8, -9, +10}), -6, -12, -20, -30, -5, +6, -7, +8));

__m128i test_mm_maskz_mullo_epi16 (__mmask8 __U, __m128i __A, __m128i __B) {
  //CHECK-LABEL: test_mm_maskz_mullo_epi16
  //CHECK: mul <8 x i16> %{{.*}}, %{{.*}}
  //CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mullo_epi16(__U , __A, __B);
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_mullo_epi16(0x0F, (__m128i)(__v8hi){+2, -3, +4, -5, +6, -7, +8, -9}, (__m128i)(__v8hi){-3, +4, -5, +6, -7, +8, -9, +10}), -6, -12, -20, -30, 0, 0, 0, 0));


__m128i test_mm_mask_blend_epi8(__mmask16 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: test_mm_mask_blend_epi8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_blend_epi8(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_blend_epi8((__mmask16)0x0001,(__m128i)(__v16qi){2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},(__m128i)(__v16qi){10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}),10,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2));

__m256i test_mm256_mask_blend_epi8(__mmask32 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: test_mm256_mask_blend_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_blend_epi8(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_blend_epi8((__mmask32)0x00000001,(__m256i)(__v32qi){2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},(__m256i)(__v32qi){10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}),10,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2));

__m128i test_mm_mask_blend_epi16(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: test_mm_mask_blend_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_blend_epi16(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_blend_epi16((__mmask8)0x01,(__m128i)(__v8hi){2,2,2,2,2,2,2,2},(__m128i)(__v8hi){10,11,12,13,14,15,16,17}),10,2,2,2,2,2,2,2));

__m256i test_mm256_mask_blend_epi16(__mmask16 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: test_mm256_mask_blend_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_blend_epi16(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_blend_epi16((__mmask16)0x0001,(__m256i)(__v16hi){2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},(__m256i)(__v16hi){10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}),10,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2));

__m128i test_mm_mask_abs_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_abs_epi8
  // CHECK: [[ABS:%.*]] = call <16 x i8> @llvm.abs.v16i8(<16 x i8> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <16 x i8> [[ABS]] to <2 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <2 x i64> [[TMP]] to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[ABS]], <16 x i8> %{{.*}}
  return _mm_mask_abs_epi8(__W,__U,__A); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_abs_epi8((__m128i)(__v16qi){99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, (__mmask16)0x0001, (__m128i)(__v16qi){(char)-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}), 1, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99));

__m128i test_mm_maskz_abs_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_abs_epi8
  // CHECK: [[ABS:%.*]] = call <16 x i8> @llvm.abs.v16i8(<16 x i8> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <16 x i8> [[ABS]] to <2 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <2 x i64> [[TMP]] to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> [[ABS]], <16 x i8> %{{.*}}
  return _mm_maskz_abs_epi8(__U,__A); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_abs_epi8((__mmask16)0x5555, (__m128i)(__v16qi){(char)-1, 2, (char)-3, 4, (char)-5, 6, (char)-7, 8, (char)-9, 10, (char)-11, 12, (char)-13, 14, (char)-15, 16}), 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0));

__m256i test_mm256_mask_abs_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_abs_epi8
  // CHECK: [[ABS:%.*]] = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <32 x i8> [[ABS]] to <4 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <4 x i64> [[TMP]] to <32 x i8>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[ABS]], <32 x i8> %{{.*}}
  return _mm256_mask_abs_epi8(__W,__U,__A); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_abs_epi8((__m256i)(__v32qi){99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, (__mmask32)0x00000001, (__m256i)(__v32qi){(char)-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}), 1, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99));

__m256i test_mm256_maskz_abs_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_abs_epi8
  // CHECK: [[ABS:%.*]] = call <32 x i8> @llvm.abs.v32i8(<32 x i8> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <32 x i8> [[ABS]] to <4 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <4 x i64> [[TMP]] to <32 x i8>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> [[ABS]], <32 x i8> %{{.*}}
  return _mm256_maskz_abs_epi8(__U,__A); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_abs_epi8((__mmask32)0x55555555, (__m256i)(__v32qi){(char)-1, 2, (char)-3, 4, (char)-5, 6, (char)-7, 8, (char)-9, 10, (char)-11, 12, (char)-13, 14, (char)-15, 16, (char)-17, 18, (char)-19, 20, (char)-21, 22, (char)-23, 24, (char)-25, 26, (char)-27, 28, (char)-29, 30, (char)-31, 32}), 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21, 0, 23, 0, 25, 0, 27, 0, 29, 0, 31, 0));

__m128i test_mm_mask_abs_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_abs_epi16
  // CHECK: [[ABS:%.*]] = call <8 x i16> @llvm.abs.v8i16(<8 x i16> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <8 x i16> [[ABS]] to <2 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <2 x i64> [[TMP]] to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> [[ABS]], <8 x i16> %{{.*}}
  return _mm_mask_abs_epi16(__W,__U,__A); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_abs_epi16((__m128i)(__v8hi){99, 99, 99, 99, 99, 99, 99, 99}, (__mmask16)0x01, (__m128i)(__v8hi){-1, 2, 2, 2, 2, 2, 2, 2}), 1, 99, 99, 99, 99, 99, 99, 99));

__m128i test_mm_maskz_abs_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_abs_epi16
  // CHECK: [[ABS:%.*]] = call <8 x i16> @llvm.abs.v8i16(<8 x i16> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <8 x i16> [[ABS]] to <2 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <2 x i64> [[TMP]] to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> [[ABS]], <8 x i16> %{{.*}}
  return _mm_maskz_abs_epi16(__U,__A); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_abs_epi16((__mmask8)0x55, (__m128i)(__v8hi){-1, 2, -3, 4, -5, 6, -7, 8}), 1, 0, 3, 0, 5, 0, 7, 0));

__m256i test_mm256_mask_abs_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_abs_epi16
  // CHECK: [[ABS:%.*]] = call <16 x i16> @llvm.abs.v16i16(<16 x i16> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <16 x i16> [[ABS]] to <4 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <4 x i64> [[TMP]] to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> [[ABS]], <16 x i16> %{{.*}}
  return _mm256_mask_abs_epi16(__W,__U,__A); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_abs_epi16((__m256i)(__v16hi){99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, (__mmask16)0x0001, (__m256i)(__v16hi){-128, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}), 128, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99));

__m256i test_mm256_maskz_abs_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_abs_epi16
  // CHECK: [[ABS:%.*]] = call <16 x i16> @llvm.abs.v16i16(<16 x i16> %{{.*}}, i1 false)
  // CHECK: [[TMP:%.*]] = bitcast <16 x i16> [[ABS]] to <4 x i64>
  // CHECK: [[ABS:%.*]] = bitcast <4 x i64> [[TMP]] to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> [[ABS]], <16 x i16> %{{.*}}
  return _mm256_maskz_abs_epi16(__U,__A); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_abs_epi16((__mmask16)0x0001, (__m256i)(__v16hi){-128, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}), 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_packs_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packs_epi32
  // CHECK: @llvm.x86.sse2.packssdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_packs_epi32(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_packs_epi32((__mmask8)0xAA,(__m128i)(__v4si){40000,-50000,65535,-65536},(__m128i)(__v4si){0,50000,40000,-40000}),0,-32768,0,-32768,0,32767,0,-32768));

__m128i test_mm_mask_packs_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packs_epi32
  // CHECK: @llvm.x86.sse2.packssdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_packs_epi32(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_packs_epi32((__m128i)(__v8hi){1,2,3,4,29,30,31,32},(__mmask8)0xAA,(__m128i)(__v4si){40000,-50000,65535,-65536},(__m128i)(__v4si){0,50000,40000,-40000}),1,-32768,3,-32768,29,32767,31,-32768));

__m256i test_mm256_maskz_packs_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packs_epi32
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_packs_epi32(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_packs_epi32((__mmask32)0xAAAA,(__m256i)(__v8si){40000,-50000,32767,-32768,32768,-32769,65535,-65536},(__m256i)(__v8si){0,1,-1,65536,22222,-22222,40000,-40000}),0,-32768,0,-32768,0,1,0,32767,0,-32768,0,-32768,0,-22222,0,-32768));

__m256i test_mm256_mask_packs_epi32(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packs_epi32
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_packs_epi32(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_packs_epi32((__m256i)(__v16hi){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xAAAA,(__m256i)(__v8si){40000,-50000,32767,-32768,32768,-32769,65535,-65536},(__m256i)(__v8si){0,1,-1,65536,22222,-22222,40000,-40000}),1,-32768,3,-32768,5,1,7,32767,25,-32768,27,-32768,29,-22222,31,-32768));

__m128i test_mm_maskz_packs_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packs_epi16
  // CHECK: @llvm.x86.sse2.packsswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_packs_epi16(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_packs_epi16((__mmask16)0xAAAA,(__m128i)(__v8hi){130,-200,127,-128,255,-255,127,-128},(__m128i)(__v8hi){0,1,-1,255,-128,90,-90,-32768}),0,-128,0,-128,0,-128,0,-128,0,1,0,127,0,90,0,-128));

__m128i test_mm_mask_packs_epi16(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packs_epi16
  // CHECK: @llvm.x86.sse2.packsswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_packs_epi16(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_packs_epi16((__m128i)(__v16qi){1,2,3,4,5,6,7,8,57,58,59,60,61,62,63,64},(__mmask16)0xAAAA,(__m128i)(__v8hi){130,-200,127,-128,255,-255,127,-128},(__m128i)(__v8hi){0,1,-1,255,-128,90,-90,-32768}),1,-128,3,-128,5,-128,7,-128,57,1,59,127,61,90,63,-128));

__m256i test_mm256_maskz_packs_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packs_epi16
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_packs_epi16(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_packs_epi16((__mmask32)0xAAAAAAAA,(__m256i)(__v16hi){130,-200,127,-128,300,-1000,42,-42,500,-500,7,-7,255,-255,127,-128},(__m256i)(__v16hi){0,1,-1,255,-129,128,20000,-32768,0,1,-1,127,-128,90,-90,-32768}),0,-128,0,-128,0,-128,0,-42,0,1,0,127,0,127,0,-128,0,-128,0,-7,0,-128,0,-128,0,1,0,127,0,90,0,-128));

__m256i test_mm256_mask_packs_epi16(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packs_epi16
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_packs_epi16(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_packs_epi16((__m256i)(__v32qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(__m256i)(__v16hi){130,-200,127,-128,300,-1000,42,-42,500,-500,7,-7,255,-255,127,-128},(__m256i)(__v16hi){0,1,-1,255,-129,128,20000,-32768,0,1,-1,127,-128,90,-90,-32768}),1,-128,3,-128,5,-128,7,-42,9,1,11,127,13,127,15,-128,49,-128,51,-7,53,-128,55,-128,57,1,59,127,61,90,63,-128));

__m128i test_mm_mask_packus_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packus_epi32
  // CHECK: @llvm.x86.sse41.packusdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_packus_epi32(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v8hu(_mm_mask_packus_epi32((__m128i)(__v8hu){1,2,3,4,5,6,7,8},(__mmask8)0xAA,(__m128i)(__v4si){40000,-50000,32767,-32768},(__m128i)(__v4si){0,1,-1,65536}),1,0,3,0,5,1,7,65535));

__m128i test_mm_maskz_packus_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packus_epi32
  // CHECK: @llvm.x86.sse41.packusdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_packus_epi32(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v8hu(_mm_maskz_packus_epi32((__mmask8)0xAA,(__m128i)(__v4si){40000,-50000,32767,-32768},(__m128i)(__v4si){0,1,-1,65536}),0,0,0,0,0,1,0,65535));

__m256i test_mm256_maskz_packus_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packus_epi32
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_packus_epi32(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_packus_epi32((__mmask16)0xAAAA,(__m256i)(__v8si){40000,-50000,32767,-32768,32768,-32769,22222,-22222},(__m256i)(__v8si){0,1,-1,65536,40000,-40000,65535,0}),0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0));

__m256i test_mm256_mask_packus_epi32(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packus_epi32
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_packus_epi32(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_packus_epi32((__m256i)(__v16hi){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xAAAA,(__m256i)(__v8si){40000,-50000,32767,-32768,32768,-32769,22222,-22222},(__m256i)(__v8si){0,1,-1,65536,40000,-40000,65535,0}),1,0,3,0,5,1,7,-1,25,0,27,0,29,0,31,0));

__m128i test_mm_maskz_packus_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packus_epi16
  // CHECK: @llvm.x86.sse2.packuswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_packus_epi16(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16qu(_mm_maskz_packus_epi16((__mmask16)0xAAAA,(__m128i)(__v8hi){-1,0,1,127,128,255,256,-200},(__m128i)(__v8hi){0,1,-1,255,-129,128,20000,-32768}),0,0,0,127,0,255,0,0,0,1,0,255,0,128,0,0));

__m128i test_mm_mask_packus_epi16(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packus_epi16
  // CHECK: @llvm.x86.sse2.packuswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_packus_epi16(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v16qu(_mm_mask_packus_epi16((__m128i)(__v16qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},(__mmask16)0xAAAA,(__m128i)(__v8hi){-1,0,1,127,128,255,256,-200},(__m128i)(__v8hi){0,1,-1,255,-129,128,20000,-32768}),1,0,3,127,5,255,7,0,9,1,11,255,13,128,15,0));

__m256i test_mm256_maskz_packus_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packus_epi16
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_packus_epi16(__M,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_packus_epi16((__mmask32)0xAAAAAAAA,(__m256i)(__v16hi){-1,0,1,127,128,255,256,-200,300,42,-42,500,20000,-32768,129,-129},(__m256i)(__v16hi){0,1,-1,255,-129,128,20000,-32768,32767,-32767,127,-128,30000,-30000,90,-90}),0,0,0,127,0,-1,0,0,0,1,0,-1,0,-128,0,0,0,42,0,-1,0,0,0,0,0,0,0,0,0,0,0,0));

__m256i test_mm256_mask_packus_epi16(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packus_epi16
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_packus_epi16(__W,__M,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_packus_epi16((__m256i)(__v32qi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(__m256i)(__v16hi){-1,0,1,127,128,255,256,-200,300,42,-42,500,20000,-32768,129,-129},(__m256i)(__v16hi){0,1,-1,255,-129,128,20000,-32768,32767,-32767,127,-128,30000,-30000,90,-90}),1,0,3,127,5,-1,7,0,9,1,11,-1,13,-128,15,0,49,42,51,-1,53,0,55,0,57,0,59,0,61,0,63,0));

__m128i test_mm_mask_adds_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epi8
  // CHECK: @llvm.sadd.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_adds_epi8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_adds_epi8((__m128i)(__v16qs){1,2,3,4,5,6,7,8,57,58,59,60,61,62,63,64},(__mmask16)0xAAAA,(__m128i)(__v16qs){0,+1,-2,+3,-4,+5,-6,+7,-100,-50,+100,-20,-80,+120,-120,-20},(__m128i)(__v16qs){0,+1,-2,+3,-4,+5,-6,+7,+50,+80,-50,+110,+60,120,+20,-120}),1,+2,3,+6,5,+10,7,+14,57,+30,59,+90,61,+127,63,-128));

__m128i test_mm_maskz_adds_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epi8
  // CHECK: @llvm.sadd.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_adds_epi8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_adds_epi8((__mmask16)0xAAAA,(__m128i)(__v16qs){0,+1,-2,+3,-4,+5,-6,+7,-100,-50,+100,-20,-80,+120,-120,-20},(__m128i)(__v16qs){0,+1,-2,+3,-4,+5,-6,+7,+50,+80,-50,+110,+60,120,+20,-120}),0,+2,0,+6,0,+10,0,+14,0,+30,0,+90,0,+127,0,-128));

__m256i test_mm256_mask_adds_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epi8
  // CHECK: @llvm.sadd.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_adds_epi8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_adds_epi8((__m256i)(__v32qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(__m256i)(__v32qs){0,+1,-2,+3,-4,+5,-6,+7,-8,+9,-10,+11,-12,+13,-14,+15,+100,+50,-100,+20,+80,-50,+120,-20,-100,-50,+100,-20,-80,+50,-120,+20},(__m256i)(__v32qs){0,+1,-2,+3,-4,+5,-6,+7,-8,+9,-10,+11,-12,+13,-14,+15,+50,+80,-50,+110,+60,-30,+20,-10,+50,+80,-50,+110,+60,-30,+20,-10}),1,+2,3,+6,5,+10,7,+14,9,+18,11,+22,13,+26,15,+30,49,+127,51,+127,53,-80,+55,-30,57,+30,59,+90,61,+20,63,+10));

__m256i test_mm256_maskz_adds_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epi8
  // CHECK: @llvm.sadd.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_adds_epi8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_adds_epi8((__mmask32)0xAAAAAAAA,(__m256i)(__v32qs){0,+1,-2,+3,-4,+5,-6,+7,-8,+9,-10,+11,-12,+13,-14,+15,+100,+50,-100,+20,+80,-50,+120,-20,-100,-50,+100,-20,-80,+50,-120,+20},(__m256i)(__v32qs){0,+1,-2,+3,-4,+5,-6,+7,-8,+9,-10,+11,-12,+13,-14,+15,+50,+80,-50,+110,+60,-30,+20,-10,+50,+80,-50,+110,+60,-30,+20,-10}),0,+2,0,+6,0,+10,0,+14,0,+18,0,+22,0,+26,0,+30,0,+127,0,+127,0,-80,0,-30,0,+30,0,+90,0,+20,0,+10));

__m128i test_mm_mask_adds_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epi16
  // CHECK: @llvm.sadd.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_adds_epi16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_adds_epi16((__m128i)(__v8hi){9,10,11,12,13,14,15,16,},(__mmask8)0xAA,(__m128i)(__v8hi){-24,+25,-26,+27,+32000,-32000,+32000,+32000},(__m128i)(__v8hi){-24,+25,-26,+27,+800,-800,-800,+800}),9,+50,11,+54,13,-32768,15,+32767));

__m128i test_mm_maskz_adds_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epi16
  // CHECK: @llvm.sadd.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_adds_epi16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_adds_epi16((__mmask8)0xAA,(__m128i)(__v8hi){-24,+25,-26,+27,+32000,-32000,+32000,+32000},(__m128i)(__v8hi){-24,+25,-26,+27,+800,-800,-800,+800}),0,+50,0,+54,0,-32768,0,+32767));

__m256i test_mm256_mask_adds_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epi16
  // CHECK: @llvm.sadd.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_adds_epi16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_adds_epi16((__m256i)(__v16hi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,},(__mmask16)0xAAAA,(__m256i)(__v16hi){0,+1,-2,+3,-4,+5,-6,+7,-24,+25,-26,+27,+32000,-32000,+32000,+32000},(__m256i)(__v16hi){0,+1,-2,+3,-4,+5,-6,+7,-24,+25,-26,+27,+800,-800,-800,+800}),1,+2,3,+6,5,+10,7,+14,9,+50,11,+54,13,-32768,15,+32767));

__m256i test_mm256_maskz_adds_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epi16
  // CHECK: @llvm.sadd.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_adds_epi16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_adds_epi16((__mmask16)0xAAAA,(__m256i)(__v16hi){0,+1,-2,+3,-4,+5,-6,+7,-24,+25,-26,+27,+32000,-32000,+32000,+32000},(__m256i)(__v16hi){0,+1,-2,+3,-4,+5,-6,+7,-24,+25,-26,+27,+800,-800,-800,+800}),0,+2,0,+6,0,+10,0,+14,0,+50,0,+54,0,-32768,0,+32767));

__m128i test_mm_mask_adds_epu8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epu8
  // CHECK-NOT: @llvm.x86.sse2.paddus.b
  // CHECK: call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_adds_epu8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qu(_mm_mask_adds_epu8((__m128i)(__v16qu){1,2,3,4,5,6,7,8,57,58,59,60,61,62,63,64},(__mmask16)0xAAAA,(__m128i)(__v16qu){0,0,0,0,0,0,0,0,+255,+255,+255,+255,+255,+255,+255,+255},(__m128i)(__v16qu){0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255}),1,+63,3,+127,5,+191,7,+255,57,+255,59,+255,61,+255,63,+255));

__m128i test_mm_maskz_adds_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epu8
  // CHECK-NOT: @llvm.x86.sse2.paddus.b
  // CHECK: call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_adds_epu8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qu(_mm_maskz_adds_epu8((__mmask16)0xAAAA,(__m128i)(__v16qu){0,0,0,0,0,0,0,0,+255,+255,+255,+255,+255,+255,+255,+255},(__m128i)(__v16qu){0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255}),0,+63,0,+127,0,+191,0,+255,0,+255,0,+255,0,+255,0,+255));

__m256i test_mm256_mask_adds_epu8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epu8
  // CHECK-NOT: @llvm.x86.avx2.paddus.b
  // CHECK: call <32 x i8> @llvm.uadd.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_adds_epu8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qu(_mm256_mask_adds_epu8((__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(__m256i)(__v32qu){0,0,0,0,0,0,0,0,+63,+63,+63,+63,+63,+63,+63,+63,+192,+192,+192,+192,+192,+192,+192,+192,+255,+255,+255,+255,+255,+255,+255,+255},(__m256i)(__v32qu){0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255}),1,+63,3,+127,5,+191,7,+255,9,+126,11,+190,13,+254,15,+255,49,+255,51,+255,53,+255,55,+255,57,+255,59,+255,61,+255,63,+255));

__m256i test_mm256_maskz_adds_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epu8
  // CHECK-NOT: @llvm.x86.avx2.paddus.b
  // CHECK: call <32 x i8> @llvm.uadd.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_adds_epu8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qu(_mm256_maskz_adds_epu8((__mmask32)0xAAAAAAAA,(__m256i)(__v32qu){0,0,0,0,0,0,0,0,+63,+63,+63,+63,+63,+63,+63,+63,+192,+192,+192,+192,+192,+192,+192,+192,+255,+255,+255,+255,+255,+255,+255,+255},(__m256i)(__v32qu){0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255,0,+63,+64,+127,+128,+191,+192,+255}),0,+63,0,+127,0,+191,0,+255,0,+126,0,+190,0,+254,0,+255,0,+255,0,+255,0,+255,0,+255,0,+255,0,+255,0,+255,0,+255));

__m128i test_mm_mask_adds_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epu16
  // CHECK-NOT: @llvm.x86.sse2.paddus.w
  // CHECK: call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_adds_epu16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hu(_mm_mask_adds_epu16((__m128i)(__v8hu){25,26,27,28,29,30,31,32},(__mmask8)0xAA,(__m128i)(__v8hu){+16384,+16384,+16384,+16384,+49152,+49152,+49152,+49152},(__m128i)(__v8hu){0,+16384,+32767,+32768,+32767,+32768,+49152,+65535}),25,+32768,27,+49152,29,+65535,31,+65535));

__m128i test_mm_maskz_adds_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epu16
  // CHECK-NOT: @llvm.x86.sse2.paddus.w
  // CHECK: call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_adds_epu16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hu(_mm_maskz_adds_epu16((__mmask8)0xAA,(__m128i)(__v8hu){+16384,+16384,+16384,+16384,+49152,+49152,+49152,+49152},(__m128i)(__v8hu){0,+16384,+32767,+32768,+32767,+32768,+49152,+65535}),0,+32768,0,+49152,0,+65535,0,+65535));

__m256i test_mm256_mask_adds_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epu16
  // CHECK-NOT: @llvm.x86.avx2.paddus.w
  // CHECK: call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_adds_epu16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hu(_mm256_mask_adds_epu16((__m256i)(__v16hu){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xAAAA,(__m256i)(__v16hu){0,0,0,0,+16384,+16384,+16384,+16384,+49152,+49152,+49152,+49152,+65535,+65535,+65535,+65535},(__m256i)(__v16hu){0,+32767,+32768,+65535,0,+16384,+32767,+32768,+32767,+32768,+49152,+65535,0,+32767,+32768,+65535}),1,+32767,3,+65535,5,+32768,7,+49152,25,+65535,27,+65535,29,+65535,31,+65535));

__m256i test_mm256_maskz_adds_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epu16
  // CHECK-NOT: @llvm.x86.avx2.paddus.w
  // CHECK: call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_adds_epu16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hu(_mm256_maskz_adds_epu16((__mmask16)0xAAAA,(__m256i)(__v16hu){0,0,0,0,+16384,+16384,+16384,+16384,+49152,+49152,+49152,+49152,+65535,+65535,+65535,+65535},(__m256i)(__v16hu){0,+32767,+32768,+65535,0,+16384,+32767,+32768,+32767,+32768,+49152,+65535,0,+32767,+32768,+65535}),0,+32767,0,+65535,0,+32768,0,+49152,0,+65535,0,+65535,0,+65535,0,+65535));

__m128i test_mm_mask_avg_epu8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_avg_epu8
  // CHECK: @llvm.x86.sse2.pavg.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_avg_epu8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_avg_epu8((__m128i)(__v16qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m128i)(__v16qi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m128i)(__v16qi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_avg_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_avg_epu8
  // CHECK: @llvm.x86.sse2.pavg.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_avg_epu8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_avg_epu8(0x00FF, (__m128i)(__v16qi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m128i)(__v16qi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_avg_epu8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_avg_epu8
  // CHECK: @llvm.x86.avx2.pavg.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_avg_epu8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_avg_epu8((__m256i)(__v32qi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0x0000FFFF, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_avg_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_avg_epu8
  // CHECK: @llvm.x86.avx2.pavg.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_avg_epu8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_avg_epu8(0x0000FFFF, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_avg_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_avg_epu16
  // CHECK: @llvm.x86.sse2.pavg.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_avg_epu16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_avg_epu16((__m128i)(__v8hi){0, 1, 2, 3, 0, 0, 0, 0}, 0x0F, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}), 1, 2, 3, 4, 0, 0, 0, 0));

__m128i test_mm_maskz_avg_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_avg_epu16
  // CHECK: @llvm.x86.sse2.pavg.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_avg_epu16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_avg_epu16(0x0F, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}), 1, 2, 3, 4, 0, 0, 0, 0));

__m256i test_mm256_mask_avg_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_avg_epu16
  // CHECK: @llvm.x86.avx2.pavg.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_avg_epu16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_avg_epu16((__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_avg_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_avg_epu16
  // CHECK: @llvm.x86.avx2.pavg.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_avg_epu16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_avg_epu16(0x00FF, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_max_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_max_epi8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_max_epi8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qi(_mm_maskz_max_epi8(0x00FF, (__m128i)(__v16qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, (__m128i)(__v16qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}), +1, +2, +3, +4, +5, +6, +7, +8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_max_epi8(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_max_epi8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_max_epi8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qi(_mm_mask_max_epi8((__m128i)(__v16qs){+1, +2, +3, +4, +5, +6, +7, +8, -9, -10, -11, -12, -13, -14, -15, -16}, 0x00FF, (__m128i)(__v16qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, (__m128i)(__v16qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}), +1, +2, +3, +4, +5, +6, +7, +8, -9, -10, -11, -12, -13, -14, -15, -16));

__m256i test_mm256_maskz_max_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_max_epi8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.smax.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_max_epi8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qi(_mm256_maskz_max_epi8(0x0000FFFF, (__m256i)(__v32qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m256i)(__v32qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18, -19, +20, -21, +22, -23, +24, -25, +26, -27, +28, -29, +30, -31, +32}), +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_max_epi8(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_max_epi8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.smax.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_max_epi8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qi(_mm256_mask_max_epi8((__m256i)(__v32qs){+1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32}, 0x0000FFFF, (__m256i)(__v32qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m256i)(__v32qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18, -19, +20, -21, +22, -23, +24, -25, +26, -27, +28, -29, +30, -31, +32}), +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32));

__m128i test_mm_maskz_max_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_max_epi16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.smax.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_max_epi16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hi(_mm_maskz_max_epi16(0x0F, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){1, 2, 3, 5, 8, 12, 20, 32}), 1, 2, 3, 5, 0, 0, 0, 0));

__m128i test_mm_mask_max_epi16(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_max_epi16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.smax.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_max_epi16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_max_epi16((__m128i)(__v8hi){1, 1, 1, 1, 0, 0, 0, 0}, 0x0F, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){1, 2, 3, 5, 8, 12, 20, 32}), 1, 2, 3, 5, 0, 0, 0, 0));

__m256i test_mm256_maskz_max_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_max_epi16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.smax.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_max_epi16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_maskz_max_epi16(0x00FF, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}), +1, +2, +3, +4, +5, +6, +7, +8, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_max_epi16(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_max_epi16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.smax.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_max_epi16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_max_epi16((__m256i)(__v16hi){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}), +1, +2, +3, +4, +5, +6, +7, +8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_max_epu8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_max_epu8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.umax.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_max_epu8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qu(_mm_maskz_max_epu8(0x00FF, (__m128i)(__v16qu){9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_max_epu8(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_max_epu8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.umax.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_max_epu8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qu(_mm_mask_max_epu8((__m128i)(__v16qu){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m128i)(__v16qu){9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_max_epu8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_max_epu8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.umax.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_max_epu8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qu(_mm256_maskz_max_epu8(0x0000FFFF, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_max_epu8(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_max_epu8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.umax.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_max_epu8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qu(_mm256_mask_max_epu8((__m256i)(__v32qu){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0x0000FFFF, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_max_epu16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_max_epu16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.umax.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_max_epu16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hu(_mm_maskz_max_epu16(0x0F, (__m128i)(__v8hu){1, 3, 5, 7, 9, 11, 13, 15}, (__m128i)(__v8hu){3, 4, 5, 6, 7, 8, 9, 10}), 3, 4, 5, 7, 0, 0, 0, 0));

__m128i test_mm_mask_max_epu16(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_max_epu16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.umax.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_max_epu16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hu(_mm_mask_max_epu16((__m128i)(__v8hu){1, 1, 1, 1, 0, 0, 0, 0}, 0x0F, (__m128i)(__v8hu){1, 3, 5, 7, 9, 11, 13, 15}, (__m128i)(__v8hu){3, 4, 5, 6, 7, 8, 9, 10}), 3, 4, 5, 7, 0, 0, 0, 0));

__m256i test_mm256_maskz_max_epu16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_max_epu16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.umax.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_max_epu16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hu(_mm256_maskz_max_epu16(0x00FF, (__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_max_epu16(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_max_epu16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.umax.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_max_epu16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hu(_mm256_mask_max_epu16((__m256i)(__v16hu){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_min_epi8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_min_epi8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_min_epi8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qi(_mm_maskz_min_epi8(0x00FF, (__m128i)(__v16qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, (__m128i)(__v16qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}), -1, -2, -3, -4, -5, -6, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_min_epi8(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_min_epi8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_min_epi8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qi(_mm_mask_min_epi8((__m128i)(__v16qs){+1, +2, +3, +4, +5, +6, +7, +8, -9, -10, -11, -12, -13, -14, -15, -16}, 0x00FF, (__m128i)(__v16qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}, (__m128i)(__v16qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}), -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16));

__m256i test_mm256_maskz_min_epi8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_min_epi8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.smin.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_min_epi8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qi(_mm256_maskz_min_epi8(0x0000FFFF, (__m256i)(__v32qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m256i)(__v32qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18, -19, +20, -21, +22, -23, +24, -25, +26, -27, +28, -29, +30, -31, +32}), -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_min_epi8(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_min_epi8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.smin.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_min_epi8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qi(_mm256_mask_min_epi8((__m256i)(__v32qs){+1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13, +14, +15, +16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32}, 0x0000FFFF, (__m256i)(__v32qs){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16, +17, -18, +19, -20, +21, -22, +23, -24, +25, -26, +27, -28, +29, -30, +31, -32}, (__m256i)(__v32qs){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16, -17, +18, -19, +20, -21, +22, -23, +24, -25, +26, -27, +28, -29, +30, -31, +32}), -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32));

__m128i test_mm_maskz_min_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_min_epi16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.smin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_min_epi16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hi(_mm_maskz_min_epi16(0x0F, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){1, 2, 3, 5, 8, 12, 20, 32}), 1, 2, 3, 4, 0, 0, 0, 0));

__m128i test_mm_mask_min_epi16(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_min_epi16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.smin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_min_epi16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_min_epi16((__m128i)(__v8hi){1, 1, 1, 1, 0, 0, 0, 0}, 0x0F, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){1, 2, 3, 5, 8, 12, 20, 32}), 1, 2, 3, 4, 0, 0, 0, 0));

__m256i test_mm256_maskz_min_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_min_epi16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.smin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_min_epi16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_maskz_min_epi16(0x00FF, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}), -1, -2, -3, -4, -5, -6, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_min_epi16(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_min_epi16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.smin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_min_epi16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_min_epi16((__m256i)(__v16hi){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-1, +2, -3, +4, -5, +6, -7, +8, -9, +10, -11, +12, -13, +14, -15, +16}), -1, -2, -3, -4, -5, -6, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_min_epu8(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_min_epu8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.umin.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_maskz_min_epu8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qu(_mm_maskz_min_epu8(0x00FF, (__m128i)(__v16qu){9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_min_epu8(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_min_epu8
  // CHECK: [[RES:%.*]] = call <16 x i8> @llvm.umin.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i8>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i8> [[RES]], <16 x i8> {{.*}}
  return _mm_mask_min_epu8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16qu(_mm_mask_min_epu8((__m128i)(__v16qu){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m128i)(__v16qu){9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_min_epu8(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_min_epu8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.umin.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_maskz_min_epu8(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qu(_mm256_maskz_min_epu8(0x0000FFFF, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_min_epu8(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_min_epu8
  // CHECK: [[RES:%.*]] = call <32 x i8> @llvm.umin.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<32 x i8>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <32 x i1> {{.*}}, <32 x i8> [[RES]], <32 x i8> {{.*}}
  return _mm256_mask_min_epu8(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v32qu(_mm256_mask_min_epu8((__m256i)(__v32qu){1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 0x0000FFFF, (__m256i)(__v32qu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}, (__m256i)(__v32qu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_maskz_min_epu16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_min_epu16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.umin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_maskz_min_epu16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hu(_mm_maskz_min_epu16(0x0F, (__m128i)(__v8hu){1, 3, 5, 7, 9, 11, 13, 15}, (__m128i)(__v8hu){3, 4, 5, 6, 7, 8, 9, 10}), 1, 3, 5, 6, 0, 0, 0, 0));

__m128i test_mm_mask_min_epu16(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_min_epu16
  // CHECK: [[RES:%.*]] = call <8 x i16> @llvm.umin.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<8 x i16>]] [[RES]] to [[DSTTY:<2 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <8 x i1> {{.*}}, <8 x i16> [[RES]], <8 x i16> {{.*}}
  return _mm_mask_min_epu16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v8hu(_mm_mask_min_epu16((__m128i)(__v8hu){1, 1, 1, 1, 0, 0, 0, 0}, 0x0F, (__m128i)(__v8hu){1, 3, 5, 7, 9, 11, 13, 15}, (__m128i)(__v8hu){3, 4, 5, 6, 7, 8, 9, 10}), 1, 3, 5, 6, 0, 0, 0, 0));

__m256i test_mm256_maskz_min_epu16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_min_epu16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.umin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_maskz_min_epu16(__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hu(_mm256_maskz_min_epu16(0x00FF, (__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_min_epu16(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_min_epu16
  // CHECK: [[RES:%.*]] = call <16 x i16> @llvm.umin.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: [[TMP:%.*]] = bitcast [[SRCTY:<16 x i16>]] [[RES]] to [[DSTTY:<4 x i64>]]
  // CHECK: [[RES:%.*]] = bitcast [[DSTTY]] [[TMP]] to [[SRCTY]]
  // CHECK:       select <16 x i1> {{.*}}, <16 x i16> [[RES]], <16 x i16> {{.*}}
  return _mm256_mask_min_epu16(__W,__M,__A,__B); 
}

TEST_CONSTEXPR(match_v16hu(_mm256_mask_min_epu16((__m256i)(__v16hu){1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 0x00FF, (__m256i)(__v16hu){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hu){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}), 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_shuffle_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_shuffle_epi8
  // CHECK: @llvm.x86.ssse3.pshuf.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_shuffle_epi8(__W,__U,__A,__B); 
}

TEST_CONSTEXPR(match_v16qi(_mm_mask_shuffle_epi8((__m128i)(__v16qi){1,1,1,1,1,1,1,1,2,2,4,4,6,6,8,8}, 0x00FF, (__m128i)(__v16qi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, (__m128i)(__v16qi){15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}), 15,14,13,12,11,10,9,8,2,2,4,4,6,6,8,8));

__m128i test_mm_maskz_shuffle_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_shuffle_epi8
  // CHECK: @llvm.x86.ssse3.pshuf.b
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_shuffle_epi8(__U,__A,__B); 
}

TEST_CONSTEXPR(match_v16qi(_mm_maskz_shuffle_epi8(0xAAAA, (__m128i)(__v16qi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, (__m128i)(__v16qi){15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}), 0,14,0,12,0,10,0,8,0,6,0,4,0,2,0,0));

__m256i test_mm256_mask_shuffle_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_shuffle_epi8
  // CHECK: @llvm.x86.avx2.pshuf.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_shuffle_epi8(__W,__U,__A,__B); 
}

TEST_CONSTEXPR(match_v32qi(_mm256_mask_shuffle_epi8((__m256i)(__v32qi){1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4}, 0x80808080, (__m256i)(__v32qi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}, (__m256i)(__v32qi){31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}), 1,1,1,1,1,1,1,8,2,2,2,2,2,2,2,0,3,3,3,3,3,3,3,24,4,4,4,4,4,4,4,16));
                

__m256i test_mm256_maskz_shuffle_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_shuffle_epi8
  // CHECK: @llvm.x86.avx2.pshuf.b
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_shuffle_epi8(__U,__A,__B); 
}

TEST_CONSTEXPR(match_v32qi(_mm256_maskz_shuffle_epi8(0x0000FFFF, (__m256i)(__v32qi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}, (__m256i)(__v32qi){31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}), 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0));

__m128i test_mm_mask_subs_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epi8
  // CHECK: @llvm.ssub.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_subs_epi8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_subs_epi8((__m128i)(__v16qs){1,2,3,4,5,6,7,8,57,58,59,60,61,62,63,64},(__mmask16)0xAAAA,(__m128i)(__v16qs){1,-100,3,4,5,-6,7,100,57,-100,59,60,61,-62,63,100},(__m128i)(__v16qs){1,100,3,4,5,6,7,-100,57,100,59,60,61,62,63,-100}),1,-128,3,0,5,-12,7,127,57,-128,59,0,61,-124,63,127));

__m128i test_mm_maskz_subs_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epi8
  // CHECK: @llvm.ssub.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_subs_epi8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_subs_epi8((__mmask16)0xAAAA,(__m128i)(__v16qs){1,-100,3,4,5,-6,7,100,57,-100,59,60,61,-62,63,100},(__m128i)(__v16qs){1,100,3,4,5,6,7,-100,57,100,59,60,61,62,63,-100}),0,-128,0,0,0,-12,0,127,0,-128,0,0,0,-124,0,127));

__m256i test_mm256_mask_subs_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epi8
  // CHECK: @llvm.ssub.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_subs_epi8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_subs_epi8((__m256i)(__v32qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(__m256i)(__v32qs){1,-100,3,4,5,-6,7,100,9,-100,11,12,13,-14,15,100,49,-100,51,52,53,-54,55,100,57,-100,59,60,61,-62,63,100},(__m256i)(__v32qs){1,100,3,4,5,6,7,-100,9,100,11,12,13,14,15,-100,49,100,51,52,53,54,55,-100,57,100,59,60,61,62,63,-100}),1,-128,3,0,5,-12,7,127,9,-128,11,0,13,-28,15,127,49,-128,51,0,53,-108,55,127,57,-128,59,0,61,-124,63,127));

__m256i test_mm256_maskz_subs_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epi8
  // CHECK: @llvm.ssub.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_subs_epi8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_subs_epi8((__mmask32)0xAAAAAAAA,(__m256i)(__v32qs){1,-100,3,4,5,-6,7,100,9,-100,11,12,13,-14,15,100,49,-100,51,52,53,-54,55,100,57,-100,59,60,61,-62,63,100},(__m256i)(__v32qs){1,100,3,4,5,6,7,-100,9,100,11,12,13,14,15,-100,49,100,51,52,53,54,55,-100,57,100,59,60,61,62,63,-100}),0,-128,0,0,0,-12,0,127,0,-128,0,0,0,-28,0,127,0,-128,0,0,0,-108,0,127,0,-128,0,0,0,-124,0,127));

__m128i test_mm_mask_subs_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epi16
  // CHECK: @llvm.ssub.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_subs_epi16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_subs_epi16((__m128i)(__v8hi){1,2,3,4,29,30,31,32},(__mmask8)0xAA,(__m128i)(__v8hi){1,-30000,3,30000,29,-30,31,32},(__m128i)(__v8hi){1,30000,3,-30000,29,30,31,-32}),1,-32768,3,32767,29,-60,31,64));

__m128i test_mm_maskz_subs_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epi16
  // CHECK: @llvm.ssub.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_subs_epi16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_subs_epi16((__mmask8)0xAA,(__m128i)(__v8hi){1,-30000,3,30000,29,-30,31,32},(__m128i)(__v8hi){1,30000,3,-30000,29,30,31,-32}),0,-32768,0,32767,0,-60,0,64));

__m256i test_mm256_mask_subs_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epi16
  // CHECK: @llvm.ssub.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_subs_epi16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_subs_epi16((__m256i)(__v16hi){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xAAAA,(__m256i)(__v16hi){1,-30000,3,30000,5,-6,7,8,25,-30000,27,30000,29,-30,31,32},(__m256i)(__v16hi){1,30000,3,-30000,5,6,7,-8,25,30000,27,-30000,29,30,31,-32}),1,-32768,3,32767,5,-12,7,16,25,-32768,27,32767,29,-60,31,64));

__m256i test_mm256_maskz_subs_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epi16
  // CHECK: @llvm.ssub.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_subs_epi16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_subs_epi16((__mmask16)0xAAAA,(__m256i)(__v16hi){1,-30000,3,30000,5,-6,7,8,25,-30000,27,30000,29,-30,31,32},(__m256i)(__v16hi){1,30000,3,-30000,5,6,7,-8,25,30000,27,-30000,29,30,31,-32}),0,-32768,0,32767,0,-12,0,16,0,-32768,0,32767,0,-60,0,64));

__m128i test_mm_mask_subs_epu8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epu8
  // CHECK-NOT: @llvm.x86.sse2.psubus.b
  // CHECK: call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_subs_epu8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qu(_mm_mask_subs_epu8((__m128i)(__v16qu){1,2,3,4,5,6,7,8,57,58,59,60,61,62,63,64},(__mmask16)0xAAAA,(__m128i)(__v16qu){0,250,0,128,0,20,0,255,0,0,0,1,0,100,0,255},(__m128i)(__v16qu){0,50,0,128,0,30,0,1,0,1,0,0,0,99,0,255}),1,200,3,0,5,0,7,254,57,0,59,1,61,1,63,0));

__m128i test_mm_maskz_subs_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epu8
  // CHECK-NOT: @llvm.x86.sse2.psubus.b
  // CHECK: call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_subs_epu8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16qu(_mm_maskz_subs_epu8((__mmask16)0xAAAA,(__m128i)(__v16qu){0,250,0,128,0,20,0,255,0,0,0,1,0,100,0,255},(__m128i)(__v16qu){0,50,0,128,0,30,0,1,0,1,0,0,0,99,0,255}),0,200,0,0,0,0,0,254,0,0,0,1,0,1,0,0));

__m256i test_mm256_mask_subs_epu8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epu8
  // CHECK-NOT: @llvm.x86.avx2.psubus.b
  // CHECK: call <32 x i8> @llvm.usub.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_subs_epu8(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qu(_mm256_mask_subs_epu8((__m256i)(__v32qu){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(__m256i)(__v32qu){0,250,0,128,0,20,0,255,0,0,0,1,0,100,0,255,0,250,0,128,0,20,0,255,0,0,0,1,0,100,0,255},(__m256i)(__v32qu){0,50,0,128,0,30,0,1,0,1,0,0,0,99,0,255,0,50,0,128,0,30,0,1,0,1,0,0,0,99,0,255}),1,200,3,0,5,0,7,254,9,0,11,1,13,1,15,0,49,200,51,0,53,0,55,254,57,0,59,1,61,1,63,0));

__m256i test_mm256_maskz_subs_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epu8
  // CHECK-NOT: @llvm.x86.avx2.psubus.b
  // CHECK: call <32 x i8> @llvm.usub.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_subs_epu8(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v32qu(_mm256_maskz_subs_epu8((__mmask32)0xAAAAAAAA,(__m256i)(__v32qu){0,250,0,128,0,20,0,255,0,0,0,1,0,100,0,255,0,250,0,128,0,20,0,255,0,0,0,1,0,100,0,255},(__m256i)(__v32qu){0,50,0,128,0,30,0,1,0,1,0,0,0,99,0,255,0,50,0,128,0,30,0,1,0,1,0,0,0,99,0,255}),0,200,0,0,0,0,0,254,0,0,0,1,0,1,0,0,0,200,0,0,0,0,0,254,0,0,0,1,0,1,0,0));

__m128i test_mm_mask_subs_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epu16
  // CHECK-NOT: @llvm.x86.sse2.psubus.w
  // CHECK: call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_subs_epu16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hu(_mm_mask_subs_epu16((__m128i)(__v8hu){101,102,103,104,129,130,131,132},(__mmask8)0xAAu,(__m128i)(__v8hu){0,65000,0,40000,0,1,0,50000},(__m128i)(__v8hu){0,5000,0,60000,0,0,0,25000}),101,60000,103,0,129,1,131,25000));

__m128i test_mm_maskz_subs_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epu16
  // CHECK-NOT: @llvm.x86.sse2.psubus.w
  // CHECK: call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_subs_epu16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v8hu(_mm_maskz_subs_epu16((__mmask8)0xAAu,(__m128i)(__v8hu){0,65000,0,40000,0,1,0,50000},(__m128i)(__v8hu){0,5000,0,60000,0,0,0,25000}),0,60000,0,0,0,1,0,25000));

__m256i test_mm256_mask_subs_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epu16
  // CHECK-NOT: @llvm.x86.avx2.psubus.w
  // CHECK: call <16 x i16> @llvm.usub.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_subs_epu16(__W,__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hu(_mm256_mask_subs_epu16((__m256i)(__v16hu){101,102,103,104,105,106,107,108,125,126,127,128,129,130,131,132},(__mmask16)0xAAAAu,(__m256i)(__v16hu){0,65000,0,40000,0,100,0,65535,0,0,0,1000,0,1,0,50000},(__m256i)(__v16hu){0,5000,0,40000,0,200,0,1,0,1,0,65535,0,0,0,25000}),101,60000,103,0,105,0,107,65534,125,0,127,0,129,1,131,25000));

__m256i test_mm256_maskz_subs_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epu16
  // CHECK-NOT: @llvm.x86.avx2.psubus.w
  // CHECK: call <16 x i16> @llvm.usub.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_subs_epu16(__U,__A,__B); 
}
TEST_CONSTEXPR(match_v16hu(_mm256_maskz_subs_epu16((__mmask16)0xAAAAu,(__m256i)(__v16hu){0,65000,0,40000,0,100,10,65535,0,0,0,1000,0,1,10000,50000},(__m256i)(__v16hu){0,5000,0,40000,0,200,0,1,0,1,0,65535,0,0,0,25000}),0,60000,0,0,0,0,0,65534,0,0,0,0,0,1,0,25000));

__m128i test_mm_mask2_permutex2var_epi16(__m128i __A, __m128i __I, __mmask8 __U, __m128i __B) {
  // CHECK-LABEL: test_mm_mask2_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask2_permutex2var_epi16(__A,__I,__U,__B); 
}
__m256i test_mm256_mask2_permutex2var_epi16(__m256i __A, __m256i __I, __mmask16 __U, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask2_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask2_permutex2var_epi16(__A,__I,__U,__B); 
}
__m128i test_mm_permutex2var_epi16(__m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: test_mm_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.128
  return _mm_permutex2var_epi16(__A,__I,__B); 
}
__m128i test_mm_mask_permutex2var_epi16(__m128i __A, __mmask8 __U, __m128i __I, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_permutex2var_epi16(__A,__U,__I,__B); 
}
__m128i test_mm_maskz_permutex2var_epi16(__mmask8 __U, __m128i __A, __m128i __I, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_permutex2var_epi16(__U,__A,__I,__B); 
}

__m256i test_mm256_permutex2var_epi16(__m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: test_mm256_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.256
  return _mm256_permutex2var_epi16(__A,__I,__B); 
}
__m256i test_mm256_mask_permutex2var_epi16(__m256i __A, __mmask16 __U, __m256i __I, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_permutex2var_epi16(__A,__U,__I,__B); 
}
__m256i test_mm256_maskz_permutex2var_epi16(__mmask16 __U, __m256i __A, __m256i __I, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_permutex2var_epi16
  // CHECK: @llvm.x86.avx512.vpermi2var.hi.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_permutex2var_epi16(__U,__A,__I,__B); 
}

TEST_CONSTEXPR(match_v8hi(
    _mm_permutex2var_epi16((__m128i)(__v8hi){0, 10, 20, 30, 40, 50, 60, 70},
                           (__m128i)(__v8hi){0, 7, 8, 15, 1, 9, 2, 10},
                           (__m128i)(__v8hi){100, 110, 120, 130, 140, 150, 160,
                                             170}),
    0, 70, 100, 170, 10, 110, 20, 120));
TEST_CONSTEXPR(match_v8hi(
    _mm_mask_permutex2var_epi16((__m128i)(__v8hi){-1, -2, -3, -4, -5, -6, -7, -8},
                                0xAA,
                                (__m128i)(__v8hi){0, 7, 8, 15, 1, 9, 2, 10},
                                (__m128i)(__v8hi){100, 110, 120, 130, 140, 150,
                                                  160, 170}),
    -1, -8, -3, 170, -5, 110, -7, 120));
TEST_CONSTEXPR(match_v8hi(
    _mm_maskz_permutex2var_epi16(0xAA,
                                 (__m128i)(__v8hi){0, 10, 20, 30, 40, 50, 60, 70},
                                 (__m128i)(__v8hi){0, 7, 8, 15, 1, 9, 2, 10},
                                 (__m128i)(__v8hi){100, 110, 120, 130, 140, 150,
                                                   160, 170}),
    0, 70, 0, 170, 0, 110, 0, 120));
TEST_CONSTEXPR(match_v8hi(
    _mm_mask2_permutex2var_epi16((__m128i)(__v8hi){0, 10, 20, 30, 40, 50, 60, 70},
                                 (__m128i)(__v8hi){0, 7, 8, 15, 1, 9, 2, 10},
                                 0x55,
                                 (__m128i)(__v8hi){100, 110, 120, 130, 140, 150,
                                                   160, 170}),
    0, 7, 100, 15, 10, 9, 20, 10));
TEST_CONSTEXPR(match_v16hi(
    _mm256_permutex2var_epi16(
        (__m256i)(__v16hi){0, 10, 20, 30, 40, 50, 60, 70,
                           80, 90, 100, 110, 120, 130, 140, 150},
        (__m256i)(__v16hi){0, 15, 16, 31, 1, 17, 2, 18,
                           3, 19, 4, 20, 5, 21, 6, 22},
        (__m256i)(__v16hi){200, 210, 220, 230, 240, 250, 260, 270,
                           280, 290, 300, 310, 320, 330, 340, 350}),
    0, 150, 200, 350, 10, 210, 20, 220,
    30, 230, 40, 240, 50, 250, 60, 260));
TEST_CONSTEXPR(match_v16hi(
    _mm256_mask_permutex2var_epi16(
        (__m256i)(__v16hi){-1, -2, -3, -4, -5, -6, -7, -8,
                           -9, -10, -11, -12, -13, -14, -15, -16},
        0xAAAA,
        (__m256i)(__v16hi){0, 15, 16, 31, 1, 17, 2, 18,
                           3, 19, 4, 20, 5, 21, 6, 22},
        (__m256i)(__v16hi){200, 210, 220, 230, 240, 250, 260, 270,
                           280, 290, 300, 310, 320, 330, 340, 350}),
    -1, -16, -3, 350, -5, 210, -7, 220,
    -9, 230, -11, 240, -13, 250, -15, 260));
TEST_CONSTEXPR(match_v16hi(
    _mm256_maskz_permutex2var_epi16(
        0x5555,
        (__m256i)(__v16hi){0, 10, 20, 30, 40, 50, 60, 70,
                           80, 90, 100, 110, 120, 130, 140, 150},
        (__m256i)(__v16hi){0, 15, 16, 31, 1, 17, 2, 18,
                           3, 19, 4, 20, 5, 21, 6, 22},
        (__m256i)(__v16hi){200, 210, 220, 230, 240, 250, 260, 270,
                           280, 290, 300, 310, 320, 330, 340, 350}),
    0, 0, 200, 0, 10, 0, 20, 0,
    30, 0, 40, 0, 50, 0, 60, 0));

__m128i test_mm_mask_maddubs_epi16(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_mask_maddubs_epi16
  // CHECK: @llvm.x86.ssse3.pmadd.ub.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_maddubs_epi16(__W, __U, __X, __Y); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_maddubs_epi16((__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, 0x0F, (__m128i)(__v16qi){1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qs){2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -2, -2, -3, -3, -4, -4}), 5, 18, 39, 68, 5, 6, 7, 8));

__m128i test_mm_maskz_maddubs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_maskz_maddubs_epi16
  // CHECK: @llvm.x86.ssse3.pmadd.ub.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_maddubs_epi16(__U, __X, __Y); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_maddubs_epi16(0x0F, (__m128i)(__v16qi){1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v16qs){2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -2, -2, -3, -3, -4, -4}), 5, 18, 39, 68, 0, 0, 0, 0));

__m256i test_mm256_mask_maddubs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_mask_maddubs_epi16
  // CHECK: @llvm.x86.avx2.pmadd.ub.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_maddubs_epi16(__W, __U, __X, __Y); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_maddubs_epi16((__m256i)(__v16hi){-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16}, 0x00FF, (__m256i)(__v32qi){1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7,8}, (__m256i)(__v32qs){2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -8, -8}), 5, 18, 39, 68, 15, 42, 77, 120, -9, -10, -11, -12, -13, -14, -15, -16));

__m256i test_mm256_maskz_maddubs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_maskz_maddubs_epi16
  // CHECK: @llvm.x86.avx2.pmadd.ub.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_maddubs_epi16(__U, __X, __Y); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_maddubs_epi16(0x00FF, (__m256i)(__v32qi){1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7,8}, (__m256i)(__v32qs){2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -8, -8}), 5, 18, 39, 68, 15, 42, 77, 120, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_madd_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_madd_epi16
  // CHECK: @llvm.x86.sse2.pmadd.wd
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_madd_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v4si(_mm_mask_madd_epi16((__m128i)(__v4si){1, 2, 3, 4}, 0x3, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){9, 10, 11, 12, 13, 14, 15, 16}), 29, 81, 3, 4));

__m128i test_mm_maskz_madd_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_madd_epi16
  // CHECK: @llvm.x86.sse2.pmadd.wd
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_madd_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v4si(_mm_maskz_madd_epi16(0x3, (__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}, (__m128i)(__v8hi){9, 10, 11, 12, 13, 14, 15, 16}), 29, 81, 0, 0));

__m256i test_mm256_mask_madd_epi16(__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_madd_epi16
  // CHECK: @llvm.x86.avx2.pmadd.wd
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_madd_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8si(_mm256_mask_madd_epi16((__m256i)(__v8si){1, 2, 3, 4, 5, 6, 7, 8}, 0x0F, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hi){10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}), 50, 250, 610, 1130, 5, 6, 7, 8));

__m256i test_mm256_maskz_madd_epi16(__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_madd_epi16
  // CHECK: @llvm.x86.avx2.pmadd.wd
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_madd_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8si(_mm256_maskz_madd_epi16(0x0F, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, (__m256i)(__v16hi){10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160}), 50, 250, 610, 1130, 0, 0, 0, 0));

__m128i test_mm_cvtsepi16_epi8(__m128i __A) {
  // CHECK-LABEL: test_mm_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.128
  return _mm_cvtsepi16_epi8(__A); 
}

__m128i test_mm_mask_cvtsepi16_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.128
  return _mm_mask_cvtsepi16_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtsepi16_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.128
  return _mm_maskz_cvtsepi16_epi8(__M, __A); 
}

__m128i test_mm256_cvtsepi16_epi8(__m256i __A) {
  // CHECK-LABEL: test_mm256_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.256
  return _mm256_cvtsepi16_epi8(__A); 
}

__m128i test_mm256_mask_cvtsepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.256
  return _mm256_mask_cvtsepi16_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtsepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtsepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.wb.256
  return _mm256_maskz_cvtsepi16_epi8(__M, __A); 
}

__m128i test_mm_cvtusepi16_epi8(__m128i __A) {
  // CHECK-LABEL: test_mm_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.128
  return _mm_cvtusepi16_epi8(__A); 
}

__m128i test_mm_mask_cvtusepi16_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.128
  return _mm_mask_cvtusepi16_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtusepi16_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.128
  return _mm_maskz_cvtusepi16_epi8(__M, __A); 
}

__m128i test_mm256_cvtusepi16_epi8(__m256i __A) {
  // CHECK-LABEL: test_mm256_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.256
  return _mm256_cvtusepi16_epi8(__A); 
}

__m128i test_mm256_mask_cvtusepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.256
  return _mm256_mask_cvtusepi16_epi8(__O, __M, __A); 
}

__m128i test_mm256_maskz_cvtusepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtusepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.wb.256
  return _mm256_maskz_cvtusepi16_epi8(__M, __A); 
}

__m128i test_mm_cvtepi16_epi8(__m128i __A) {
  // CHECK-LABEL: test_mm_cvtepi16_epi8
  // CHECK: trunc <8 x i16> %{{.*}} to <8 x i8>
  // CHECK: shufflevector <8 x i8> %{{.*}}, <8 x i8> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  return _mm_cvtepi16_epi8(__A); 
}

TEST_CONSTEXPR(match_v16qi(_mm_cvtepi16_epi8((__m128i)(__v8hi){1, 2, 3, 4, 5, 6, 7, 8}), 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_cvtepi16_epi8(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.128
  return _mm_mask_cvtepi16_epi8(__O, __M, __A); 
}

__m128i test_mm_maskz_cvtepi16_epi8(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtepi16_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.wb.128
  return _mm_maskz_cvtepi16_epi8(__M, __A); 
}

__m128i test_mm256_cvtepi16_epi8(__m256i __A) {
  // CHECK-LABEL: test_mm256_cvtepi16_epi8
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  return _mm256_cvtepi16_epi8(__A); 
}

TEST_CONSTEXPR(match_v16qi(_mm256_cvtepi16_epi8((__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));

__m128i test_mm256_mask_cvtepi16_epi8(__m128i __O, __mmask16 __M, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtepi16_epi8
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm256_mask_cvtepi16_epi8(__O, __M, __A); 
}

TEST_CONSTEXPR(match_v16qi(_mm256_mask_cvtepi16_epi8(_mm_set1_epi8(-177), /*1010 0011 0011 0101=*/0xa335, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, -177, 3, -177, 5, 6, -177, -177, 9, 10, -177, -177, -177, 14, -177, 16));

__m128i test_mm256_maskz_cvtepi16_epi8(__mmask16 __M, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtepi16_epi8
  // CHECK: trunc <16 x i16> %{{.*}} to <16 x i8>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm256_maskz_cvtepi16_epi8(__M, __A); 
}

TEST_CONSTEXPR(match_v16qi(_mm256_maskz_cvtepi16_epi8(/*1010 0011 0011 0101=*/0xa335, (__m256i)(__v16hi){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), 1, 0, 3, 0, 5, 6, 0, 0, 9, 10, 0, 0, 0, 14, 0, 16));

__m128i test_mm_mask_mulhrs_epi16(__m128i __W, __mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_mask_mulhrs_epi16
  // CHECK: @llvm.x86.ssse3.pmul.hr.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_mulhrs_epi16(_mm_set1_epi16(1), 0x0F, (__m128i)(__v8hi){+100, +200, -300, -400, +500, +600, -700, +800}, (__m128i)(__v8hi){+8000, -7000, +6000, -5000, +4000, -3000, +2000, -1000}), +24, -43, -55, +61, +1, +1, +1, +1));

__m128i test_mm_maskz_mulhrs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.ssse3.pmul.hr.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhrs_epi16(__U, __X, __Y); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_mulhrs_epi16(0x0F, (__m128i)(__v8hi){+100, +200, -300, -400, +500, +600, -700, +800}, (__m128i)(__v8hi){+8000, -7000, +6000, -5000, +4000, -3000, +2000, -1000}), +24, -43, -55, +61, 0, 0, 0, 0));

__m256i test_mm256_mask_mulhrs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_mask_mulhrs_epi16
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_mulhrs_epi16(_mm256_set1_epi16(1), 0xF00F, (__m256i)(__v16hi){+100, +200, -300, -400, +500, +600, -700, +800, -900, -1000, +1100, +1200, -1300, -1400, +1500, +1600}, (__m256i)(__v16hi){+1600, -1500, +1400, -1300, +1200, -1100, +1000, -900, +800, -700, +600, -500, +400, -300, +200, -100}), +5, -9, -13, +16, +1, +1, +1, +1, +1, +1, +1, +1, -16, +13, +9, -5));

__m256i test_mm256_maskz_mulhrs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhrs_epi16(__U, __X, __Y); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_mulhrs_epi16(0xF00F, (__m256i)(__v16hi){+100, +200, -300, -400, +500, +600, -700, +800, -900, -1000, +1100, +1200, -1300, -1400, +1500, +1600}, (__m256i)(__v16hi){+1600, -1500, +1400, -1300, +1200, -1100, +1000, -900, +800, -700, +600, -500, +400, -300, +200, -100}), +5, -9, -13, +16, 0, 0, 0, 0, 0, 0, 0, 0, -16, +13, +9, -5));

__m128i test_mm_mask_mulhi_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_mulhi_epu16
  // CHECK: @llvm.x86.sse2.pmulhu.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mulhi_epu16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_mulhi_epu16(_mm_set1_epi16(1), 0x3C, (__m128i)(__v8hi){+1, -2, +3, -4, +5, -6, +7, -8}, (__m128i)(__v8hi){-16, -14, +12, +10, -8, +6, -4, +2}), 1, 1, 0, 9, 4, 5, 1, 1));

__m128i test_mm_maskz_mulhi_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_mulhi_epu16
  // CHECK: @llvm.x86.sse2.pmulhu.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhi_epu16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_mulhi_epu16(0x87, (__m128i)(__v8hi){+1, -2, +3, -4, +5, -6, +7, -8}, (__m128i)(__v8hi){-16, -14, +12, +10, -8, +6, -4, +2}), 0, -16, 0, 0, 0, 0, 0, 1));

__m256i test_mm256_mask_mulhi_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_mulhi_epu16
  // CHECK: @llvm.x86.avx2.pmulhu.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhi_epu16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_mulhi_epu16(_mm256_set1_epi16(1), 0xF00F, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), 0, -32, 0, 25, 1, 1, 1, 1, 1, 1, 1, 1, 12, 5, 14, 1));

__m256i test_mm256_maskz_mulhi_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_mulhi_epu16
  // CHECK: @llvm.x86.avx2.pmulhu.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhi_epu16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_mulhi_epu16(0x0FF0, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), 0, 0, 0, 0, 4, -28, 0, 17, 8, -24, 0, 9, 0, 0, 0, 0));

__m128i test_mm_mask_mulhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_mulhi_epi16
  // CHECK: @llvm.x86.sse2.pmulh.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mulhi_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_mulhi_epu16(_mm_set1_epi16(1), 0x78, (__m128i)(__v8hi){+1, -2, +3, -4, +5, -6, +7, -8}, (__m128i)(__v8hi){-16, -14, +12, +10, -8, +6, -4, +2}), 1, 1, 1, 9, 4, 5, 6, 1));

__m128i test_mm_maskz_mulhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_mulhi_epi16
  // CHECK: @llvm.x86.sse2.pmulh.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhi_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_mulhi_epi16(0xC3, (__m128i)(__v8hi){+1, -2, +3, -4, +5, -6, +7, -8}, (__m128i)(__v8hi){-16, -14, +12, +10, -8, +6, -4, +2}), -1, 0, 0, 0, 0, 0, -1, -1));

__m256i test_mm256_mask_mulhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_mulhi_epi16
  // CHECK: @llvm.x86.avx2.pmulh.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhi_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_mulhi_epi16(_mm256_set1_epi16(1), 0x0FF0, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), 1, 1, 1, 1, -1, 0, 0, -1, -1, 0, 0, -1, 1, 1, 1, 1));

__m256i test_mm256_maskz_mulhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_mulhi_epi16
  // CHECK: @llvm.x86.avx2.pmulh.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhi_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_mulhi_epi16(0xF00F, (__m256i)(__v16hi){+1, -2, +3, -4, +5, -6, +7, -8, +9, -10, +11, -12, +13, -14, +15, -16}, (__m256i)(__v16hi){-32, -30, +28, +26, -24, -22, +20, +18, -16, -14, +12, +10, -8, +6, -4, +2}), -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1));

__m128i test_mm_mask_unpackhi_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpackhi_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_unpackhi_epi8(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_unpackhi_epi8((__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},(__mmask16)0xFAAA,(__m128i)(__v16qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115},(__m128i)(__v16qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}),1,-9,3,-10,5,-11,7,-12,9,-13,11,-14,114,-15,115,-16));

__m128i test_mm_maskz_unpackhi_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpackhi_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpackhi_epi8(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_unpackhi_epi8((__mmask16)0xFAAA,(__m128i)(__v16qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115},(__m128i)(__v16qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}),0,-9,0,-10,0,-11,0,-12,0,-13,0,-14,114,-15,115,-16));

__m256i test_mm256_mask_unpackhi_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpackhi_epi8(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_unpackhi_epi8((__m256i)(__v32qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xFAAAAAAA,(__m256i)(__v32qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,-104,-103,-102,-101,-100,-99,-98,-97,-96,-95,-94,-93,-92,-91,-90,-89},(__m256i)(__v32qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-49,-50,-51,-52,-53,-54,-55,-56,-57,-58,-59,-60,-61,-62,-63,-64}),1,-9,3,-10,5,-11,7,-12,9,-13,11,-14,13,-15,15,-16,49,-57,51,-58,53,-59,55,-60,57,-61,59,-62,-90,-63,-89,-64));

__m256i test_mm256_maskz_unpackhi_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpackhi_epi8(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_unpackhi_epi8((__mmask32)0xFAAAAAAA,(__m256i)(__v32qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,-104,-103,-102,-101,-100,-99,-98,-97,-96,-95,-94,-93,-92,-91,-90,-89},(__m256i)(__v32qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-49,-50,-51,-52,-53,-54,-55,-56,-57,-58,-59,-60,-61,-62,-63,-64}),0,-9,0,-10,0,-11,0,-12,0,-13,0,-14,0,-15,0,-16,0,-57,0,-58,0,-59,0,-60,0,-61,0,-62,-90,-63,-89,-64));

__m128i test_mm_mask_unpackhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_unpackhi_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_unpackhi_epi16((__m128i)(__v8hi){1,2,3,4,5,6,7,8},(__mmask8)0xFA,(__m128i)(__v8hi){100,101,102,103,104,105,106,107},(__m128i)(__v8hi){200,201,202,203,204,205,206,207}),1,204,3,205,106,206,107,207));

__m128i test_mm_maskz_unpackhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_unpackhi_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_unpackhi_epi16((__mmask8)0xFA,(__m128i)(__v8hi){100,101,102,103,104,105,106,107},(__m128i)(__v8hi){200,201,202,203,204,205,206,207}),0,204,0,205,106,206,107,207));

__m256i test_mm256_mask_unpackhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_unpackhi_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_unpackhi_epi16((__m256i)(__v16hi){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xFAAAu,(__m256i)(__v16hi){100,101,102,103,104,105,106,107,130,131,132,133,134,135,136,137},(__m256i)(__v16hi){200,201,202,203,204,205,206,207,230,231,232,233,234,235,236,237}),1,204,3,205,5,206,7,207,25,234,27,235,136,236,137,237));

__m256i test_mm256_maskz_unpackhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_unpackhi_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_unpackhi_epi16((__mmask16)0xFAAAu,(__m256i)(__v16hi){100,101,102,103,104,105,106,107,130,131,132,133,134,135,136,137},(__m256i)(__v16hi){200,201,202,203,204,205,206,207,230,231,232,233,234,235,236,237}),0,204,0,205,0,206,0,207,0,234,0,235,136,236,137,237));

__m128i test_mm_mask_unpacklo_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_unpacklo_epi8(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_unpacklo_epi8((__m128i)(__v16qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},(__mmask16)0xFAAA,(__m128i)(__v16qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115},(__m128i)(__v16qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}),1,-1,3,-2,5,-3,7,-4,9,-5,11,-6,106,-7,107,-8));

__m128i test_mm_maskz_unpacklo_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpacklo_epi8(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_unpacklo_epi8((__mmask16)0xFAAA,(__m128i)(__v16qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115},(__m128i)(__v16qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16}),0,-1,0,-2,0,-3,0,-4,0,-5,0,-6,106,-7,107,-8));

__m256i test_mm256_mask_unpacklo_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpacklo_epi8(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_unpacklo_epi8((__m256i)(__v32qs){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xFAAAAAAA,(__m256i)(__v32qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,-50,-51,-52,-53,-54,-55,-56,-57,-58,-59,-60,-61,-62,-63,-64,-65},(__m256i)(__v32qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75}),1,-1,3,-2,5,-3,7,-4,9,-5,11,-6,13,-7,15,-8,49,60,51,61,53,62,55,63,57,64,59,65,-56,66,-57,67));

__m256i test_mm256_maskz_unpacklo_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpacklo_epi8(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_unpacklo_epi8((__mmask32)0xFAAAAAAA,(__m256i)(__v32qs){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,-50,-51,-52,-53,-54,-55,-56,-57,-58,-59,-60,-61,-62,-63,-64,-65},(__m256i)(__v32qs){-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75}),0,-1,0,-2,0,-3,0,-4,0,-5,0,-6,0,-7,0,-8,0,60,0,61,0,62,0,63,0,64,0,65,-56,66,-57,67));

__m128i test_mm_mask_unpacklo_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_unpacklo_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_unpacklo_epi16((__m128i)(__v8hi){1,2,3,4,5,6,7,8},(__mmask8)0xFAu,(__m128i)(__v8hi){100,101,102,103,104,105,106,107},(__m128i)(__v8hi){200,201,202,203,204,205,206,207}),1,200,3,201,102,202,103,203));

__m128i test_mm_maskz_unpacklo_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_unpacklo_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_unpacklo_epi16((__mmask8)0xFAu,(__m128i)(__v8hi){100,101,102,103,104,105,106,107},(__m128i)(__v8hi){200,201,202,203,204,205,206,207}),0,200,0,201,102,202,103,203));

__m256i test_mm256_mask_unpacklo_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_unpacklo_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_unpacklo_epi16((__m256i)(__v16hi){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xFAAAu,(__m256i)(__v16hi){100,101,102,103,104,105,106,107,130,131,132,133,134,135,136,137},(__m256i)(__v16hi){200,201,202,203,204,205,206,207,230,231,232,233,234,235,236,237}),1,200,3,201,5,202,7,203,25,230,27,231,132,232,133,233));

__m256i test_mm256_maskz_unpacklo_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_unpacklo_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_unpacklo_epi16((__mmask16)0xFAAAu,(__m256i)(__v16hi){100,101,102,103,104,105,106,107,130,131,132,133,134,135,136,137},(__m256i)(__v16hi){200,201,202,203,204,205,206,207,230,231,232,233,234,235,236,237}),0,200,0,201,0,202,0,203,0,230,0,231,132,232,133,233));

__m128i test_mm_mask_cvtepi8_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtepi8_epi16
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_cvtepi8_epi16(__W, __U, __A); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_cvtepi8_epi16(_mm_set1_epi16(-777),(__mmask8)0xA5,(__m128i)(__v16qs){1,-2,3,-4,5,-6,7,-8,9,10,11,12,13,14,15,16}),1,-777,3,-777,-777,-6,-777,-8));

__m128i test_mm_maskz_cvtepi8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtepi8_epi16
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_cvtepi8_epi16(__U, __A); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_cvtepi8_epi16((__mmask8)0xA5,(__m128i)(__v16qs){1,-2,3,-4,5,-6,7,-8,9,10,11,12,13,14,15,16}),1,0,3,0,0,-6,0,-8));

__m256i test_mm256_mask_cvtepi8_epi16(__m256i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_cvtepi8_epi16(__W, __U, __A); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_cvtepi8_epi16(_mm256_set1_epi16(-777),/*1001110010100101=*/0x9ca5,(__m128i)(__v16qs){1,-2,3,-4,5,-6,7,-8,25,-26,27,-28,29,-30,31,-32}),1,-777,3,-777,-777,-6,-777,-8,-777,-777,27,-28,29,-777,-777,-32));

__m256i test_mm256_maskz_cvtepi8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_cvtepi8_epi16(__U, __A); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_cvtepi8_epi16(/*1001110010100101=*/0x9ca5,(__m128i)(__v16qs){1,-2,3,-4,5,-6,7,-8,25,-26,27,-28,29,-30,31,-32}),1,0,3,0,0,-6,0,-8,0,0,27,-28,29,0,0,-32));

__m128i test_mm_mask_cvtepu8_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtepu8_epi16
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_cvtepu8_epi16(__W, __U, __A); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_cvtepu8_epi16(_mm_set1_epi16(-777),(__mmask8)0xA5,(__m128i)(__v16qu){25,26,27,28,29,30,31,32,0,0,0,0,0,0,0,0}),25,-777,27,-777,-777,30,-777,32));

__m128i test_mm_maskz_cvtepu8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtepu8_epi16
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_cvtepu8_epi16(__U, __A); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_cvtepu8_epi16((__mmask8)0xA5,(__m128i)(__v16qu){25,26,27,28,29,30,31,32,0,0,0,0,0,0,0,0}),25,0,27,0,0,30,0,32));

__m256i test_mm256_mask_cvtepu8_epi16(__m256i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_cvtepu8_epi16(__W, __U, __A); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_cvtepu8_epi16(_mm256_set1_epi16(-777),/*1001110010100101=*/0x9ca5,(__m128i)(__v16qu){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32}),1,-777,3,-777,-777,6,-777,8,-777,-777,27,28,29,-777,-777,32));

__m256i test_mm256_maskz_cvtepu8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_cvtepu8_epi16(__U, __A); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_cvtepu8_epi16(/*1001110010100101=*/0x9ca5,(__m128i)(__v16qu){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32}),1,0,3,0,0,6,0,8,0,0,27,28,29,0,0,32));

__m256i test_mm256_sllv_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.256(
  return _mm256_sllv_epi16(__A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_sllv_epi16((__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}),  -64, 0, -272, 560, -1152, -2368, 0, -9984, 0, 0, 20480, 0, -32768, 0, 0, 0));

__m256i test_mm256_mask_sllv_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sllv_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_sllv_epi16((__m256i)(__v16hi){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}, 0xFF56, (__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}), 999, 0, -272, 999, -1152, 999, 0, 999, 0, 0, 20480, 0, -32768, 0, 0, 0));

__m256i test_mm256_maskz_sllv_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sllv_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_sllv_epi16(0xFF56, (__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}),  0, 0, -272, 0, -1152, 0, 0, 0, 0, 0, 20480, 0, -32768, 0, 0, 0));

__m128i test_mm_sllv_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.128(
  return _mm_sllv_epi16(__A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_sllv_epi16((__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}),  32, 68, 0, 0, -640, 0, 0, 5888));

__m128i test_mm_mask_sllv_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sllv_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_sllv_epi16((__m128i)(__v8hi){ 999, 999, 999, 999, 999, 999, 999, 999}, 0x93, (__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}), 32, 68, 999, 999, -640, 999, 999, 5888));

__m128i test_mm_maskz_sllv_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_sllv_epi16
  // CHECK: @llvm.x86.avx512.psllv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sllv_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_sllv_epi16(0x93, (__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}),  32, 68, 0, 0, -640, 0, 0, 5888));

__m128i test_mm_mask_sll_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_sll_epi16
  // CHECK: @llvm.x86.sse2.psll.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sll_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sll_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_sll_epi16
  // CHECK: @llvm.x86.sse2.psll.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sll_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_sll_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: test_mm256_mask_sll_epi16
  // CHECK: @llvm.x86.avx2.psll.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sll_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sll_epi16(__mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: test_mm256_maskz_sll_epi16
  // CHECK: @llvm.x86.avx2.psll.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sll_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_slli_epi16((__mmask8)0xAA, (__m128i)(__v8hi){0, 1, 2, 3, 4, 5, 6, 7}, 20), 0, 0, 0, 0, 0, 0, 0, 0));

__m128i test_mm_mask_slli_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_slli_epi16
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_slli_epi16(__W, __U, __A, 5); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_slli_epi16((__m128i)(__v8hi){100, 101, 102, 103, 104, 105, 106, 107}, (__mmask8)0xAA, (__m128i)(__v8hi){0, 1, 2, 3, 4, 5, 6, 7}, 20), 100, 0, 102, 0, 104, 0, 106, 0));

__m128i test_mm_mask_slli_epi16_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm_mask_slli_epi16_2
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_slli_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_slli_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_slli_epi16
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_slli_epi16(__U, __A, 5); 
}

__m128i test_mm_maskz_slli_epi16_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm_maskz_slli_epi16_2
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_slli_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_slli_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_slli_epi16
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_slli_epi16(__W, __U, __A, 5); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_slli_epi16((__m256i)(__v16hi){100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}, (__mmask16)0xAAAA, (__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 20), 100, 0, 102, 0, 104, 0, 106, 0, 108, 0, 110, 0, 112, 0, 114, 0));

__m256i test_mm256_mask_slli_epi16_2(__m256i __W, __mmask16 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm256_mask_slli_epi16_2
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_slli_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_slli_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_slli_epi16
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_slli_epi16(__U, __A, 5); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_slli_epi16((__mmask16)0x00ffcc71, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 32), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_slli_epi16((__mmask16)0, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 16), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_slli_epi16((__mmask16)0xffff, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0x1fe, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_slli_epi16((__mmask16)0x7, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0x1fe, 0x2, 0x4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_slli_epi16((__mmask16)0x71, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0x1fe, 0, 0, 0, 0x8, 0xa, 0xc, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_slli_epi16_2(__mmask16 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm256_maskz_slli_epi16_2
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_slli_epi16(__U, __A, __B); 
}

__m256i test_mm256_srlv_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.256(
  return _mm256_srlv_epi16(__A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_srlv_epi16((__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}),  32752, 0, 8187, 2, 2046, 1023, 0, 255, 0, 0, 0, 0, 0, 0, 1, 0));

__m256i test_mm256_mask_srlv_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srlv_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_srlv_epi16((__m256i)(__v16hi){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}, 0xFF56, (__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}), 999, 0, 8187, 999, 2046, 999, 0, 999, 0, 0, 0, 0, 0, 0, 1, 0));

__m256i test_mm256_maskz_srlv_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srlv_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_srlv_epi16(0xFF56, (__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}),  0, 0, 8187, 0, 2046, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0));

__m128i test_mm_srlv_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.128(
  return _mm_srlv_epi16(__A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_srlv_epi16((__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}),  8, 4, 0, 0, 2047, 0, 0, 0));

__m128i test_mm_mask_srlv_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srlv_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_srlv_epi16((__m128i)(__v8hi){ 999, 999, 999, 999, 999, 999, 999, 999}, 0x93, (__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}), 8, 4, 999, 999, 2047, 999, 999, 0));

__m128i test_mm_maskz_srlv_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_srlv_epi16
  // CHECK: @llvm.x86.avx512.psrlv.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srlv_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_srlv_epi16(0x93, (__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}),  8, 4, 0, 0, 2047, 0, 0, 0));

__m128i test_mm_mask_srl_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_srl_epi16
  // CHECK: @llvm.x86.sse2.psrl.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srl_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srl_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_srl_epi16
  // CHECK: @llvm.x86.sse2.psrl.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srl_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_srl_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: test_mm256_mask_srl_epi16
  // CHECK: @llvm.x86.avx2.psrl.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srl_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srl_epi16(__mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: test_mm256_maskz_srl_epi16
  // CHECK: @llvm.x86.avx2.psrl.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srl_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_srli_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_srli_epi16
  // CHECK: @llvm.x86.sse2.psrli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srli_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_mask_srli_epi16_2(__m128i __W, __mmask8 __U, __m128i __A, int __B) {
  // CHECK-LABEL: test_mm_mask_srli_epi16_2
  // CHECK: @llvm.x86.sse2.psrli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srli_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srli_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_srli_epi16
  // CHECK: @llvm.x86.sse2.psrli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srli_epi16(__U, __A, 5); 
}

__m128i test_mm_maskz_srli_epi16_2(__mmask8 __U, __m128i __A, int __B) {
  // CHECK-LABEL: test_mm_maskz_srli_epi16_2
  // CHECK: @llvm.x86.sse2.psrli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srli_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_srli_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_srli_epi16
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srli_epi16(__W, __U, __A, 5); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_srli_epi16((__m256i)(__v16hi){100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}, (__mmask16)0xAAAA, (__m256i)(__v16hi){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 20), 100, 0, 102, 0, 104, 0, 106, 0, 108, 0, 110, 0, 112, 0, 114, 0));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_srli_epi16((__m256i)(__v16hi){100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}, (__mmask16)0xAAAA, (__m256i)(__v16hi){0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480}, 5), 100, 1, 102, 3, 104, 5, 106, 7, 108, 9, 110, 11, 112, 13, 114, 15));

__m256i test_mm256_mask_srli_epi16_2(__m256i __W, __mmask16 __U, __m256i __A, int __B) {
  // CHECK-LABEL: test_mm256_mask_srli_epi16_2
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srli_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srli_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_srli_epi16
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srli_epi16(__U, __A, 5); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_srli_epi16((__mmask16)0x71, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0x7f, 0, 0, 0, 0x2, 0x2, 0x3, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_srli_epi16_2(__mmask16 __U, __m256i __A, int __B) {
  // CHECK-LABEL: test_mm256_maskz_srli_epi16_2
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srli_epi16(__U, __A, __B); 
}

__m256i test_mm256_srav_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.256(
  return _mm256_srav_epi16(__A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_srav_epi16((__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}),  -16, 0, -5, 2, -2, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1));

__m256i test_mm256_mask_srav_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srav_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_srav_epi16((__m256i)(__v16hi){ 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}, 0xFF56, (__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}), 999, 0, -5, 999, -2, 999, 0, 999, -1, -1, 0, 0, 0, -1, -1, -1));

__m256i test_mm256_maskz_srav_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.256(
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srav_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_srav_epi16(0xFF56, (__m256i)(__v16hi){ -32, 33, -34, 35, -36, -37, 38, -39, -40, -41, 42, 43, 44, -45, -46, -47}, (__m256i)(__v16hi){ 1, -2, 3, 4, 5, 6, -7, 8, -9, -10, 11, -12, 13, -14, 15, 16}),  0, 0, -5, 0, -2, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, -1));

__m128i test_mm_srav_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.128(
  return _mm_srav_epi16(__A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_srav_epi16((__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}),  8, 4, 0, -1, -1, -1, 0, 0));

__m128i test_mm_mask_srav_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srav_epi16(__W, __U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_srav_epi16((__m128i)(__v8hi){ 999, 999, 999, 999, 999, 999, 999, 999}, 0x93, (__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}), 8, 4, 999, 999, -1, 999, 999, 0));

__m128i test_mm_maskz_srav_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_srav_epi16
  // CHECK: @llvm.x86.avx512.psrav.w.128(
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srav_epi16(__U, __A, __B); 
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_srav_epi16(0x93, (__m128i)(__v8hi){ 16, 17, 18, -19, -20, -21, 22, 23}, (__m128i)(__v8hi){ 1, 2, -3, -4, 5, -6, -7, 8}),  8, 4, 0, 0, -1, 0, 0, 0));

__m128i test_mm_mask_sra_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_sra_epi16
  // CHECK: @llvm.x86.sse2.psra.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_sra_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_sra_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_sra_epi16
  // CHECK: @llvm.x86.sse2.psra.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_sra_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_sra_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: test_mm256_mask_sra_epi16
  // CHECK: @llvm.x86.avx2.psra.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_sra_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_sra_epi16(__mmask16 __U, __m256i __A, __m128i __B) {
  // CHECK-LABEL: test_mm256_maskz_sra_epi16
  // CHECK: @llvm.x86.avx2.psra.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_sra_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_srai_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_srai_epi16
  // CHECK: @llvm.x86.sse2.psrai.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srai_epi16(__W, __U, __A, 5); 
}

__m128i test_mm_mask_srai_epi16_2(__m128i __W, __mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm_mask_srai_epi16_2
  // CHECK: @llvm.x86.sse2.psrai.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_srai_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_srai_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_srai_epi16
  // CHECK: @llvm.x86.sse2.psrai.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srai_epi16(__U, __A, 5); 
}

__m128i test_mm_maskz_srai_epi16_2(__mmask8 __U, __m128i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm_maskz_srai_epi16_2
  // CHECK: @llvm.x86.sse2.psrai.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_srai_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_srai_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_srai_epi16
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srai_epi16(__W, __U, __A, 5); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_srai_epi16((__m256i)(__v16hi){100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}, (__mmask16)0xAAAA, (__m256i)(__v16hi){0, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 20), 100, 0Xffff, 102, 0, 104, 0, 106, 0, 108, 0, 110, 0, 112, 0, 114, 0));

__m256i test_mm256_mask_srai_epi16_2(__m256i __W, __mmask16 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm256_mask_srai_epi16_2
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_srai_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_srai_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_srai_epi16
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srai_epi16(__U, __A, 5); 
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_srai_epi16((__mmask16)0x71, (__m256i)(__v16hi){0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 1), 0x7f, 0, 0, 0, 0x2, 0x2, 0x3, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_maskz_srai_epi16_2(__mmask16 __U, __m256i __A, unsigned int __B) {
  // CHECK-LABEL: test_mm256_maskz_srai_epi16_2
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_srai_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_mov_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_mov_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_mov_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_mov_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mov_epi16(__U, __A); 
}

__m256i test_mm256_mask_mov_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_mov_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mov_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_mov_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mov_epi16(__U, __A); 
}

__m128i test_mm_mask_mov_epi8(__m128i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_mov_epi8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_mov_epi8(__W, __U, __A); 
}

__m128i test_mm_maskz_mov_epi8(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_mov_epi8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_mov_epi8(__U, __A); 
}

__m256i test_mm256_mask_mov_epi8(__m256i __W, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_mov_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_mov_epi8(__W, __U, __A); 
}

__m256i test_mm256_maskz_mov_epi8(__mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_mov_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_mov_epi8(__U, __A); 
}

__m128i test_mm_loadu_epi16(void const *__P) {
  // CHECK-LABEL: test_mm_loadu_epi16
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm_loadu_epi16(__P);
}

__m128i test_mm_mask_loadu_epi16(__m128i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm_mask_loadu_epi16
  // CHECK: @llvm.masked.load.v8i16.p0(ptr align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mask_loadu_epi16(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi16(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v8i16.p0(ptr align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_maskz_loadu_epi16(__U, __P); 
}

__m256i test_mm256_loadu_epi16(void const *__P) {
  // CHECK-LABEL: test_mm256_loadu_epi16
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm256_loadu_epi16(__P);
}

__m256i test_mm256_mask_loadu_epi16(__m256i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_mask_loadu_epi16
  // CHECK: @llvm.masked.load.v16i16.p0(ptr align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mask_loadu_epi16(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi16(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v16i16.p0(ptr align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_maskz_loadu_epi16(__U, __P); 
}

__m128i test_mm_loadu_epi8(void const *__P) {
  // CHECK-LABEL: test_mm_loadu_epi8
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm_loadu_epi8(__P);
}

__m128i test_mm_mask_loadu_epi8(__m128i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v16i8.p0(ptr align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_mask_loadu_epi8(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi8(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v16i8.p0(ptr align 1 %{{.*}}, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_maskz_loadu_epi8(__U, __P); 
}

__m256i test_mm256_loadu_epi8(void const *__P) {
  // CHECK-LABEL: test_mm256_loadu_epi8
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm256_loadu_epi8(__P);
}

__m256i test_mm256_mask_loadu_epi8(__m256i __W, __mmask32 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v32i8.p0(ptr align 1 %{{.*}}, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_mask_loadu_epi8(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi8(__mmask32 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v32i8.p0(ptr align 1 %{{.*}}, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_maskz_loadu_epi8(__U, __P); 
}

void test_mm_storeu_epi16(void *__p, __m128i __a) {
  // CHECK-LABEL: test_mm_storeu_epi16
  // check: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm_storeu_epi16(__p, __a);
}

void test_mm_mask_storeu_epi16(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v8i16.p0(<8 x i16> %{{.*}}, ptr align 1 %{{.*}}, <8 x i1> %{{.*}})
  return _mm_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm256_storeu_epi16(void *__P, __m256i __A) {
  // CHECK-LABEL: test_mm256_storeu_epi16
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm256_storeu_epi16(__P, __A);
}

void test_mm256_mask_storeu_epi16(void *__P, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v16i16.p0(<16 x i16> %{{.*}}, ptr align 1 %{{.*}}, <16 x i1> %{{.*}})
  return _mm256_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm_storeu_epi8(void *__p, __m128i __a) {
  // CHECK-LABEL: test_mm_storeu_epi8
  // check: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm_storeu_epi8(__p, __a);
}

void test_mm_mask_storeu_epi8(void *__P, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v16i8.p0(<16 x i8> %{{.*}}, ptr align 1 %{{.*}}, <16 x i1> %{{.*}})
  return _mm_mask_storeu_epi8(__P, __U, __A); 
}

void test_mm256_storeu_epi8(void *__P, __m256i __A) {
  // CHECK-LABEL: test_mm256_storeu_epi8
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm256_storeu_epi8(__P, __A);
}

void test_mm256_mask_storeu_epi8(void *__P, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v32i8.p0(<32 x i8> %{{.*}}, ptr align 1 %{{.*}}, <32 x i1> %{{.*}})
  return _mm256_mask_storeu_epi8(__P, __U, __A); 
}
__mmask16 test_mm_test_epi8_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_test_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return _mm_test_epi8_mask(__A, __B); 
}

TEST_CONSTEXPR(_mm_test_epi8_mask(
    (__m128i)(__v16qi){1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    (__m128i)(__v16qi){1, 2, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
)
== (__mmask16)0xfffb);

__mmask16 test_mm_mask_test_epi8_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_test_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi8_mask(__U, __A, __B); 
}
TEST_CONSTEXPR(_mm_mask_test_epi8_mask(
    0xFFFF,
    (__m128i)(__v16qi){1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    (__m128i)(__v16qi){1, 2, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
)
== (__mmask16)0xfffb);

__mmask32 test_mm256_test_epi8_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_test_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return _mm256_test_epi8_mask(__A, __B); 
}
TEST_CONSTEXPR(_mm256_test_epi8_mask(
    (__m256i)(__v32qi){1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    (__m256i)(__v32qi){1, 2, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
)
== (__mmask32)0xfffbfffb);

__mmask32 test_mm256_mask_test_epi8_mask(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_test_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_test_epi8_mask(__U, __A, __B); 
}

__mmask8 test_mm_test_epi16_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_test_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  return _mm_test_epi16_mask(__A, __B); 
}

__mmask8 test_mm_mask_test_epi16_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_test_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm256_test_epi16_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_test_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  return _mm256_test_epi16_mask(__A, __B); 
}

__mmask16 test_mm256_mask_test_epi16_mask(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_test_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_test_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm_testn_epi8_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_testn_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  return _mm_testn_epi8_mask(__A, __B); 
}

TEST_CONSTEXPR(_mm_testn_epi8_mask(
    (__m128i)(__v16qi){1, 2, 77, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 16, 16},
    (__m128i)(__v16qi){2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15}
)
== (__mmask16)0xe001);

__mmask16 test_mm_mask_testn_epi8_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_testn_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_testn_epi8_mask(__U, __A, __B); 
}

__mmask32 test_mm256_testn_epi8_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_testn_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  return _mm256_testn_epi8_mask(__A, __B); 
}

__mmask32 test_mm256_mask_testn_epi8_mask(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_testn_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <32 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <32 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_testn_epi8_mask(__U, __A, __B); 
}

__mmask8 test_mm_testn_epi16_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_testn_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  return _mm_testn_epi16_mask(__A, __B); 
}

__mmask8 test_mm_mask_testn_epi16_mask(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_testn_epi16_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <8 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <8 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_testn_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm256_testn_epi16_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_testn_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  return _mm256_testn_epi16_mask(__A, __B); 
}

__mmask16 test_mm256_mask_testn_epi16_mask(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_testn_epi16_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp eq <16 x i16> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm256_mask_testn_epi16_mask(__U, __A, __B); 
}

__mmask16 test_mm_movepi8_mask(__m128i __A) {
  // CHECK-LABEL: test_mm_movepi8_mask
  // CHECK: [[CMP:%.*]] = icmp slt <16 x i8> %{{.*}}, zeroinitializer
  return _mm_movepi8_mask(__A); 
}

TEST_CONSTEXPR(_mm_movepi8_mask(
    ((__m128i)(__v16qi){0, 1, char(129), 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})) == (__mmask16)0x0004); 

__mmask32 test_mm256_movepi8_mask(__m256i __A) {
  // CHECK-LABEL: test_mm256_movepi8_mask
  // CHECK: [[CMP:%.*]] = icmp slt <32 x i8> %{{.*}}, zeroinitializer
  return _mm256_movepi8_mask(__A); 
}

TEST_CONSTEXPR(_mm256_movepi8_mask(
    ((__m256i)(__v32qi){0, 1, char(255), 3, 4, 5, 6, 7,
                        8, 9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, char(128)})
) == (__mmask32)0x80000004);

__m128i test_mm_movm_epi8(__mmask16 __A) {
  // CHECK-LABEL: test_mm_movm_epi8
  // CHECK: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: %vpmovm2.i = sext <16 x i1> %{{.*}} to <16 x i8>
  return _mm_movm_epi8(__A); 
}

__m256i test_mm256_movm_epi8(__mmask32 __A) {
  // CHECK-LABEL: test_mm256_movm_epi8
  // CHECK: %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // CHECK: %vpmovm2.i = sext <32 x i1> %{{.*}} to <32 x i8>
  return _mm256_movm_epi8(__A); 
}

__m128i test_mm_movm_epi16(__mmask8 __A) {
  // CHECK-LABEL: test_mm_movm_epi16
  // CHECK: %{{.*}} = bitcast i8 %{{.*}} to <8 x i1>
  // CHECK: %vpmovm2.i = sext <8 x i1> %{{.*}} to <8 x i16>
  return _mm_movm_epi16(__A); 
}

__m256i test_mm256_movm_epi16(__mmask16 __A) {
  // CHECK-LABEL: test_mm256_movm_epi16
  // CHECK: %{{.*}} = bitcast i16 %{{.*}} to <16 x i1>
  // CHECK: %vpmovm2.i = sext <16 x i1> %{{.*}} to <16 x i16>
  return _mm256_movm_epi16(__A); 
}

__m128i test_mm_mask_broadcastb_epi8(__m128i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_broadcastb_epi8(__O, __M, __A);
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_broadcastb_epi8((__m128i)(__v16qs){0,1,2,3,4,5,6,7,56,57,58,59,60,61,62,63},(__mmask16)0xAAAA,(__m128i)(__v16qs){-120,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),0,-120,2,-120,4,-120,6,-120,56,-120,58,-120,60,-120,62,-120));

__m128i test_mm_maskz_broadcastb_epi8(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_broadcastb_epi8(__M, __A);
}
TEST_CONSTEXPR(match_v16qi(_mm_maskz_broadcastb_epi8((__mmask16)0xAAAA,(__m128i)(__v16qs){-120,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120));

__m256i test_mm256_mask_broadcastb_epi8(__m256i __O, __mmask32 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_broadcastb_epi8(__O, __M, __A);
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_broadcastb_epi8((__m256i)(__v32qs){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63},(__mmask32)0xAAAAAAAA,(__m128i)(__v16qs){-120,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),0,-120,2,-120,4,-120,6,-120,8,-120,10,-120,12,-120,14,-120,48,-120,50,-120,52,-120,54,-120,56,-120,58,-120,60,-120,62,-120));

__m256i test_mm256_maskz_broadcastb_epi8(__mmask32 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_broadcastb_epi8(__M, __A);
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_broadcastb_epi8((__mmask32)0xAAAAAAAA,(__m128i)(__v16qs){-120,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120));

__m128i test_mm_mask_broadcastw_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_broadcastw_epi16(__O, __M, __A);
}
TEST_CONSTEXPR(match_v8hi(_mm_mask_broadcastw_epi16((__m128i)(__v8hi){0,1,2,3,4,5,6,7},(__mmask8)0xAA,(__m128i)(__v8hi){-120,1,2,3,4,5,6,7}),0,-120,2,-120,4,-120,6,-120));

__m128i test_mm_maskz_broadcastw_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_broadcastw_epi16(__M, __A);
}
TEST_CONSTEXPR(match_v8hi(_mm_maskz_broadcastw_epi16((__mmask8)0xAA,(__m128i)(__v8hi){-120,1,2,3,4,5,6,7}),0,-120,0,-120,0,-120,0,-120));

__m256i test_mm256_mask_broadcastw_epi16(__m256i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_broadcastw_epi16(__O, __M, __A);
}
TEST_CONSTEXPR(match_v16hi(_mm256_mask_broadcastw_epi16((__m256i)(__v16hi){0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31},(__mmask16)0xAAAA,(__m128i)(__v8hi){-120,1,2,3,4,5,6,7}),0,-120,2,-120,4,-120,6,-120,24,-120,26,-120,28,-120,30,-120));

__m256i test_mm256_maskz_broadcastw_epi16(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_broadcastw_epi16(__M, __A);
}
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_broadcastw_epi16((__mmask16)0xAAAA,(__m128i)(__v8hi){-120,1,2,3,4,5,6,7}),0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120,0,-120));

__m128i test_mm_mask_set1_epi8 (__m128i __O, __mmask16 __M, char __A){
  // CHECK-LABEL: test_mm_mask_set1_epi8
  // CHECK: insertelement <16 x i8> poison, i8 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v16qi(_mm_mask_set1_epi8((__m128i)(__v16qi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},(__mmask16)0xAAAA,(char)42),1,42,3,42,5,42,7,42,9,42,11,42,13,42,15,42));

__m128i test_mm_maskz_set1_epi8 ( __mmask16 __M, char __A){
  // CHECK-LABEL: test_mm_maskz_set1_epi8
  // CHECK: insertelement <16 x i8> poison, i8 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v16qi(_mm_maskz_set1_epi8((__mmask16)0xAAAA,(char)42),0,42,0,42,0,42,0,42,0,42,0,42,0,42,0,42));

__m256i test_mm256_mask_set1_epi8(__m256i __O, __mmask32 __M, char __A) {
  // CHECK-LABEL: test_mm256_mask_set1_epi8
  // CHECK: insertelement <32 x i8> poison, i8 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v32qi(_mm256_mask_set1_epi8((__m256i)(__v32qi){1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64},(__mmask32)0xAAAAAAAA,(char)42),1,42,3,42,5,42,7,42,9,42,11,42,13,42,15,42,49,42,51,42,53,42,55,42,57,42,59,42,61,42,63,42));

__m256i test_mm256_maskz_set1_epi8( __mmask32 __M, char __A) {
  // CHECK-LABEL: test_mm256_maskz_set1_epi8
  // CHECK: insertelement <32 x i8> poison, i8 %{{.*}}, i32 0
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
TEST_CONSTEXPR( match_v32qi( _mm256_maskz_set1_epi8( (__mmask32)0xAAAAAAAA, (char)42 ), 0,   42,  0, 42, 0,  42,  0,  42, 0,   42,  0, 42, 0,  42,  0,  42, 0,   42,  0, 42, 0,  42,  0,  42, 0,   42,  0, 42, 0,  42,  0,  42 ) );

__m256i test_mm256_mask_set1_epi16(__m256i __O, __mmask16 __M, short __A) {
  // CHECK-LABEL: test_mm256_mask_set1_epi16
  // CHECK: insertelement <16 x i16> poison, i16 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v16hi(_mm256_mask_set1_epi16((__m256i)(__v16hi){1,2,3,4,5,6,7,8,25,26,27,28,29,30,31,32},(__mmask16)0xAAAA,42),1,42,3,42,5,42,7,42,25,42,27,42,29,42,31,42));

__m256i test_mm256_maskz_set1_epi16(__mmask16 __M, short __A) {
  // CHECK-LABEL: test_mm256_maskz_set1_epi16
  // CHECK: insertelement <16 x i16> poison, i16 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_set1_epi16((__mmask16)0xAAAA,42),0,42,0,42,0,42,0,42,0,42,0,42,0,42,0,42));

__m128i test_mm_mask_set1_epi16(__m128i __O, __mmask8 __M, short __A) {
  // CHECK-LABEL: test_mm_mask_set1_epi16
  // CHECK: insertelement <8 x i16> poison, i16 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v8hi(_mm_mask_set1_epi16((__m128i)(__v8hi){1,2,3,4,5,6,7,8},(__mmask8)0xAA,42),1,42,3,42,5,42,7,42));

__m128i test_mm_maskz_set1_epi16(__mmask8 __M, short __A) {
  // CHECK-LABEL: test_mm_maskz_set1_epi16
  // CHECK: insertelement <8 x i16> poison, i16 %{{.*}}, i32 0
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
TEST_CONSTEXPR(match_v8hi(_mm_maskz_set1_epi16((__mmask8)0xAA,42),0,42,0,42,0,42,0,42));

__m128i test_mm_permutexvar_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.128
  return _mm_permutexvar_epi16(__A, __B);
}

TEST_CONSTEXPR(match_v8hi(_mm_permutexvar_epi16((__m128i)(__v8hi){7, 6, 5, 4, 3, 2, 1, 0}, (__m128i)(__v8hi){10, 11, 12, 13, 14, 15, 16, 17}), 17, 16, 15, 14, 13, 12, 11, 10));
TEST_CONSTEXPR(match_v8hi(_mm_permutexvar_epi16((__m128i)(__v8hi){0, 0, 0, 0, 0, 0, 0, 0}, (__m128i)(__v8hi){100, 101, 102, 103, 104, 105, 106, 107}), 100, 100, 100, 100, 100, 100, 100, 100));

__m128i test_mm_maskz_permutexvar_epi16(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_permutexvar_epi16(__M, __A, __B); 
}

__m128i test_mm_mask_permutexvar_epi16(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_permutexvar_epi16(__W, __M, __A, __B);
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_permutexvar_epi16((__m128i)(__v8hi){99, 99, 99, 99, 99, 99, 99, 99}, 0xFF, (__m128i)(__v8hi){7, 6, 5, 4, 3, 2, 1, 0}, (__m128i)(__v8hi){10, 11, 12, 13, 14, 15, 16, 17}), 17, 16, 15, 14, 13, 12, 11, 10));
TEST_CONSTEXPR(match_v8hi(_mm_mask_permutexvar_epi16((__m128i)(__v8hi){99, 99, 99, 99, 99, 99, 99, 99}, 0xAA, (__m128i)(__v8hi){7, 6, 5, 4, 3, 2, 1, 0}, (__m128i)(__v8hi){10, 11, 12, 13, 14, 15, 16, 17}), 99, 16, 99, 14, 99, 12, 99, 10));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_permutexvar_epi16(0xFF, (__m128i)(__v8hi){7, 6, 5, 4, 3, 2, 1, 0}, (__m128i)(__v8hi){10, 11, 12, 13, 14, 15, 16, 17}), 17, 16, 15, 14, 13, 12, 11, 10));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_permutexvar_epi16(0xAA, (__m128i)(__v8hi){7, 6, 5, 4, 3, 2, 1, 0}, (__m128i)(__v8hi){10, 11, 12, 13, 14, 15, 16, 17}), 0, 16, 0, 14, 0, 12, 0, 10));

__m256i test_mm256_permutexvar_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.256
  return _mm256_permutexvar_epi16(__A, __B);
}

TEST_CONSTEXPR(match_v16hi(_mm256_permutexvar_epi16((__m256i)(__v16hi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, (__m256i)(__v16hi){10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}), 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10));
TEST_CONSTEXPR(match_v16hi(_mm256_permutexvar_epi16((__m256i)(__v16hi){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, (__m256i)(__v16hi){100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115}), 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100));

__m256i test_mm256_maskz_permutexvar_epi16(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_permutexvar_epi16(__M, __A, __B); 
}

__m256i test_mm256_mask_permutexvar_epi16(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_permutexvar_epi16(__W, __M, __A, __B);
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_permutexvar_epi16((__m256i)(__v16hi){99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0xFFFF, (__m256i)(__v16hi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, (__m256i)(__v16hi){10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}), 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_permutexvar_epi16((__m256i)(__v16hi){99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}, 0xAAAA, (__m256i)(__v16hi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, (__m256i)(__v16hi){10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}), 99, 24, 99, 22, 99, 20, 99, 18, 99, 16, 99, 14, 99, 12, 99, 10));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_permutexvar_epi16(0xFFFF, (__m256i)(__v16hi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, (__m256i)(__v16hi){10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}), 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_permutexvar_epi16(0xAAAA, (__m256i)(__v16hi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, (__m256i)(__v16hi){10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}), 0, 24, 0, 22, 0, 20, 0, 18, 0, 16, 0, 14, 0, 12, 0, 10));

__m128i test_mm_mask_alignr_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}
TEST_CONSTEXPR(match_v16qi(_mm_mask_alignr_epi8(((__m128i)(__v16qs){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}), (__mmask16)0x000f, ((__m128i)(__v16qs){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), ((__m128i)(__v16qs){17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}), 2), 19, 20, 21, 22, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127));

__m128i test_mm_maskz_alignr_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_alignr_epi8(__U, __A, __B, 2); 
}
TEST_CONSTEXPR(match_v16qi( _mm_maskz_alignr_epi8((__mmask16)0x000f, ((__m128i)(__v16qs){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}), ((__m128i)(__v16qs){17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}),2), 19, 20, 21, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_alignr_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_mask_alignr_epi8(((__m256i)(__v32qs){127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127}), (__mmask32)0xf000000f, ((__m256i)(__v32qs){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}), ((__m256i)(__v32qs){33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}), 2), 35, 36, 37, 38, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 63, 64, 17, 18));

__m256i test_mm256_maskz_alignr_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_alignr_epi8(__U, __A, __B, 2); 
}
TEST_CONSTEXPR(match_v32qi(_mm256_maskz_alignr_epi8((__mmask32)0xf000000f, ((__m256i)(__v32qs){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}), ((__m256i)(__v32qs){33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}), 2), 35, 36, 37, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 64, 17, 18));

__m128i test_mm_dbsad_epu8(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.128
  return _mm_dbsad_epu8(__A, __B, 170); 
}

__m128i test_mm_mask_dbsad_epu8(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_dbsad_epu8(__W, __U, __A, __B, 170); 
}

__m128i test_mm_maskz_dbsad_epu8(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.128
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_dbsad_epu8(__U, __A, __B, 170); 
}

__m256i test_mm256_dbsad_epu8(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.256
  return _mm256_dbsad_epu8(__A, __B, 170); 
}

__m256i test_mm256_mask_dbsad_epu8(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_dbsad_epu8(__W, __U, __A, __B, 170); 
}

__m256i test_mm256_maskz_dbsad_epu8(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_dbsad_epu8
  // CHECK: @llvm.x86.avx512.dbpsadbw.256
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_dbsad_epu8(__U, __A, __B, 170); 
}
__mmask8 test_mm_movepi16_mask(__m128i __A) {
  // CHECK-LABEL: test_mm_movepi16_mask
  // CHECK: [[CMP:%.*]] = icmp slt <8 x i16> %{{.*}}, zeroinitializer
  return _mm_movepi16_mask(__A); 
}
TEST_CONSTEXPR(_mm_movepi16_mask(((__m128i)(__v8hi){0, 1, -1, 3, 4, 5, 6, 7})) == (__mmask8)0x04);

__mmask16 test_mm256_movepi16_mask(__m256i __A) {
  // CHECK-LABEL: test_mm256_movepi16_mask
  // CHECK: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer
  return _mm256_movepi16_mask(__A); 
}

TEST_CONSTEXPR(_mm256_movepi16_mask(
    ((__m256i)(__v16hi){0, 1, short(32769), 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, short(65535)})) == (__mmask16)0x8004);

__m128i test_mm_mask_shufflehi_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_shufflehi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shufflehi_epi16(__W, __U, __A, 5); 
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflehi_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0xF0u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),100,101,102,103,5,5,4,4));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflehi_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0x00u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),100,101,102,103,104,105,106,107));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflehi_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0xFFu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,1,2,3,5,5,4,4));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflehi_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0x0Fu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,1,2,3,104,105,106,107));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflehi_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0x55u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,101,2,103,5,105,4,107));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflehi_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0xAAu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),100,1,102,3,104,5,106,4));

__m128i test_mm_maskz_shufflehi_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_shufflehi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shufflehi_epi16(__U, __A, 5); 
}

TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflehi_epi16(0xF0u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,0,0,0,5,5,4,4));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflehi_epi16(0x00u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,0,0,0,0,0,0,0));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflehi_epi16(0xFFu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,1,2,3,5,5,4,4));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflehi_epi16(0x0Fu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,1,2,3,0,0,0,0));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflehi_epi16(0x55u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,0,2,0,5,0,4,0));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflehi_epi16(0xAAu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,1,0,3,0,5,0,4));

__m128i test_mm_mask_shufflelo_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_shufflelo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_shufflelo_epi16(__W, __U, __A, 5); 
}

TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflelo_epi16(((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),0xFF,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),1,1,0,0,4,5,6,7));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflelo_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0x00u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),100,101,102,103,104,105,106,107));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflelo_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0x0Fu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),1,1,0,0,104,105,106,107));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflelo_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0xF0u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),100,101,102,103,4,5,6,7));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflelo_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0xAAu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),100,1,102,0,104,5,106,7));
TEST_CONSTEXPR(match_v8hi(_mm_mask_shufflelo_epi16(((__m128i)(__v8hi){100,101,102,103,104,105,106,107}),0x55u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),1,101,0,103,4,105,6,107));

__m128i test_mm_maskz_shufflelo_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_shufflelo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> poison, <8 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_shufflelo_epi16(__U, __A, 5); 
}

TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflelo_epi16(0xFF,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),1,1,0,0,4,5,6,7));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflelo_epi16(0x0Fu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),1,1,0,0,0,0,0,0));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflelo_epi16(0xF0u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,0,0,0,4,5,6,7));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflelo_epi16(0xAAu,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),0,1,0,0,0,5,0,7));
TEST_CONSTEXPR(match_v8hi(_mm_maskz_shufflelo_epi16(0x55u,((__m128i)(__v8hi){0,1,2,3,4,5,6,7}),5),1,0,0,0,4,0,6,0));

__m256i test_mm256_mask_shufflehi_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_shufflehi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shufflehi_epi16(__W, __U, __A, 5); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflehi_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115}),0xFF00u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),100,101,102,103,104,105,106,107,8,9,10,11,13,13,12,12));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflehi_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115}),0x0000u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflehi_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115}),0xFFFFu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,1,2,3,5,5,4,4,8,9,10,11,13,13,12,12));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflehi_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115}),0x00FFu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,1,2,3,5,5,4,4,108,109,110,111,112,113,114,115));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflehi_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115}),0x5555u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,101,2,103,5,105,4,107,8,109,10,111,13,113,12,115));

__m256i test_mm256_maskz_shufflehi_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_shufflehi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shufflehi_epi16(__U, __A, 5); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflehi_epi16(0x0000u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflehi_epi16(0xFFFFu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,1,2,3,5,5,4,4,8,9,10,11,13,13,12,12));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflehi_epi16(0x00FFu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,1,2,3,5,5,4,4,0,0,0,0,0,0,0,0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflehi_epi16(0xFF00u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,0,0,0,0,0,0,0,8,9,10,11,13,13,12,12));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflehi_epi16(0x5555u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,0,2,0,5,0,4,0,8,0,10,0,13,0,12,0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflehi_epi16(0xAAAAu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,1,0,3,0,5,0,4,0,9,0,11,0,13,0,12));

__m256i test_mm256_mask_shufflelo_epi16(__m256i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_shufflelo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_shufflelo_epi16(__W, __U, __A, 5); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflelo_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,200,201,202,203,204,205,206,207}),0xFFFF,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,4,5,6,7,9,9,8,8,12,13,14,15));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflelo_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,200,201,202,203,204,205,206,207}),0x000Fu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,104,105,106,107,200,201,202,203,204,205,206,207));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflelo_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,200,201,202,203,204,205,206,207}),0x00FFu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,4,5,6,7,200,201,202,203,204,205,206,207));
TEST_CONSTEXPR(match_v16hi(_mm256_mask_shufflelo_epi16(((__m256i)(__v16hi){100,101,102,103,104,105,106,107,200,201,202,203,204,205,206,207}),0xF00Fu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,104,105,106,107,200,201,202,203,12,13,14,15));

__m256i test_mm256_maskz_shufflelo_epi16(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_shufflelo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> poison, <16 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_shufflelo_epi16(__U, __A, 5); 
}

TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflelo_epi16(0xFFFF,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,4,5,6,7,9,9,8,8,12,13,14,15));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflelo_epi16(0x000Fu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflelo_epi16(0x00FFu,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),1,1,0,0,4,5,6,7,0,0,0,0,0,0,0,0));
TEST_CONSTEXPR(match_v16hi(_mm256_maskz_shufflelo_epi16(0xF0F0u,((__m256i)(__v16hi){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}),5),0,0,0,0,4,5,6,7,0,0,0,0,12,13,14,15));

void test_mm_mask_cvtepi16_storeu_epi8 (void * __P, __mmask8 __M, __m128i __A)
{
 // CHECK-LABEL: test_mm_mask_cvtepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmov.wb.mem.128
 _mm_mask_cvtepi16_storeu_epi8 (__P, __M, __A);
}

void test_mm_mask_cvtsepi16_storeu_epi8 (void * __P, __mmask8 __M, __m128i __A)
{
 // CHECK-LABEL: test_mm_mask_cvtsepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovs.wb.mem.128
  _mm_mask_cvtsepi16_storeu_epi8 ( __P,  __M, __A);
}

void test_mm_mask_cvtusepi16_storeu_epi8 (void * __P, __mmask8 __M, __m128i __A)
{
 // CHECK-LABEL: test_mm_mask_cvtusepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovus.wb.mem.128
  _mm_mask_cvtusepi16_storeu_epi8 (__P, __M, __A);
}

void test_mm256_mask_cvtusepi16_storeu_epi8 (void * __P, __mmask16 __M, __m256i __A)
{
 // CHECK-LABEL: test_mm256_mask_cvtusepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovus.wb.mem.256
  _mm256_mask_cvtusepi16_storeu_epi8 ( __P, __M, __A);
}

void test_mm256_mask_cvtepi16_storeu_epi8 (void * __P, __mmask16 __M, __m256i __A)
{
 // CHECK-LABEL: test_mm256_mask_cvtepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmov.wb.mem.256
  _mm256_mask_cvtepi16_storeu_epi8 ( __P,  __M, __A);
}

void test_mm256_mask_cvtsepi16_storeu_epi8 (void * __P, __mmask16 __M, __m256i __A)
{
 // CHECK-LABEL: test_mm256_mask_cvtsepi16_storeu_epi8
 // CHECK: @llvm.x86.avx512.mask.pmovs.wb.mem.256
 _mm256_mask_cvtsepi16_storeu_epi8 ( __P, __M, __A);
}


TEST_CONSTEXPR(match_v16qu(
    _mm_permutex2var_epi8((__m128i)(__v16qu){0, 10, 20, 30, 40, 50, 60, 70,
                                             80, 90, 100, 110, 120, 127, 126, 125},
                         (__m128i)(__v16qu){0, 16, 1, 17, 2, 18, 3, 19,
                                             4, 20, 5, 21, 6, 22, 7, 23},
                         (__m128i)(__v16qu){100, 110, 120, 130, 140, 150, 160, 170,
                                            180, 190, 200, 210, 220, 230, 240, 250}),
    0, 100, 10, 110, 20, 120, 30, 130,
    40, 140, 50, 150, 60, 160, 70, 170));
TEST_CONSTEXPR(match_v32qu(
    _mm256_permutex2var_epi8((__m256i)(__v32qu){0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109},
                             (__m256i)(__v32qu){0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47},
                             (__m256i)(__v32qu){200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231}),
    0, 200, 10, 201, 20, 202, 30, 203,
    40, 204, 50, 205, 60, 206, 70, 207,
    80, 208, 90, 209, 100, 210, 110, 211,
    120, 212, 127, 213, 126, 214, 125, 215));
