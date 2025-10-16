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
TEST_CONSTEXPR(match_v16qi(
  _mm_mask_blend_epi8(
    (__mmask16)0x0001,
    (__m128i)(__v16qi){2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    (__m128i)(__v16qi){ 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 }
  ),
  10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
));

__m256i test_mm256_mask_blend_epi8(__mmask32 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: test_mm256_mask_blend_epi8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_blend_epi8(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v32qi(
  _mm256_mask_blend_epi8(
    (__mmask32) 0x00000001,
    (__m256i)(__v32qi) {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    (__m256i)(__v32qi){ 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}
  ),
  10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
));

__m128i test_mm_mask_blend_epi16(__mmask8 __U, __m128i __A, __m128i __W) {
  // CHECK-LABEL: test_mm_mask_blend_epi16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_blend_epi16(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v8hi(
  _mm_mask_blend_epi16(
    (__mmask8)0x01,
    (__m128i)(__v8hi){2, 2, 2, 2, 2, 2, 2, 2},
    (__m128i)(__v8hi){ 10,11,12,13,14,15,16,17 }
  ),
  10, 2, 2, 2, 2, 2, 2, 2
));

__m256i test_mm256_mask_blend_epi16(__mmask16 __U, __m256i __A, __m256i __W) {
  // CHECK-LABEL: test_mm256_mask_blend_epi16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_blend_epi16(__U,__A,__W); 
}
TEST_CONSTEXPR(match_v16hi(
  _mm256_mask_blend_epi16(
    (__mmask16)0x0001,
    (__m256i)(__v16hi){2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    (__m256i)(__v16hi){ 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 }
  ),
  10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
));

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
__m128i test_mm_mask_packs_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packs_epi32
  // CHECK: @llvm.x86.sse2.packssdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_packs_epi32(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_packs_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packs_epi32
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_packs_epi32(__M,__A,__B); 
}
__m256i test_mm256_mask_packs_epi32(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packs_epi32
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_packs_epi32(__W,__M,__A,__B); 
}
__m128i test_mm_maskz_packs_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packs_epi16
  // CHECK: @llvm.x86.sse2.packsswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_packs_epi16(__M,__A,__B); 
}
__m128i test_mm_mask_packs_epi16(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packs_epi16
  // CHECK: @llvm.x86.sse2.packsswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_packs_epi16(__W,__M,__A,__B); 
}
__m256i test_mm256_maskz_packs_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packs_epi16
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_packs_epi16(__M,__A,__B); 
}
__m256i test_mm256_mask_packs_epi16(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packs_epi16
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_packs_epi16(__W,__M,__A,__B); 
}

__m128i test_mm_mask_packus_epi32(__m128i __W, __mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packus_epi32
  // CHECK: @llvm.x86.sse41.packusdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_packus_epi32(__W,__M,__A,__B); 
}

__m128i test_mm_maskz_packus_epi32(__mmask8 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packus_epi32
  // CHECK: @llvm.x86.sse41.packusdw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_packus_epi32(__M,__A,__B); 
}

__m256i test_mm256_maskz_packus_epi32(__mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packus_epi32
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_packus_epi32(__M,__A,__B); 
}

__m256i test_mm256_mask_packus_epi32(__m256i __W, __mmask16 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packus_epi32
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_packus_epi32(__W,__M,__A,__B); 
}

__m128i test_mm_maskz_packus_epi16(__mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_packus_epi16
  // CHECK: @llvm.x86.sse2.packuswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_packus_epi16(__M,__A,__B); 
}

__m128i test_mm_mask_packus_epi16(__m128i __W, __mmask16 __M, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_packus_epi16
  // CHECK: @llvm.x86.sse2.packuswb
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_packus_epi16(__W,__M,__A,__B); 
}

__m256i test_mm256_maskz_packus_epi16(__mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_packus_epi16
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_packus_epi16(__M,__A,__B); 
}

__m256i test_mm256_mask_packus_epi16(__m256i __W, __mmask32 __M, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_packus_epi16
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_packus_epi16(__W,__M,__A,__B); 
}

__m128i test_mm_mask_adds_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epi8
  // CHECK: @llvm.sadd.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_adds_epi8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epi8
  // CHECK: @llvm.sadd.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_adds_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epi8
  // CHECK: @llvm.sadd.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_adds_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epi8
  // CHECK: @llvm.sadd.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_adds_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epi16
  // CHECK: @llvm.sadd.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_adds_epi16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epi16
  // CHECK: @llvm.sadd.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_adds_epi16(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epi16
  // CHECK: @llvm.sadd.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_adds_epi16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epi16
  // CHECK: @llvm.sadd.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_adds_epi16(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epu8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epu8
  // CHECK-NOT: @llvm.x86.sse2.paddus.b
  // CHECK: call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_adds_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epu8
  // CHECK-NOT: @llvm.x86.sse2.paddus.b
  // CHECK: call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_adds_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epu8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epu8
  // CHECK-NOT: @llvm.x86.avx2.paddus.b
  // CHECK: call <32 x i8> @llvm.uadd.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_adds_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epu8
  // CHECK-NOT: @llvm.x86.avx2.paddus.b
  // CHECK: call <32 x i8> @llvm.uadd.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_adds_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_adds_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_adds_epu16
  // CHECK-NOT: @llvm.x86.sse2.paddus.w
  // CHECK: call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_adds_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_adds_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_adds_epu16
  // CHECK-NOT: @llvm.x86.sse2.paddus.w
  // CHECK: call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_adds_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_adds_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_adds_epu16
  // CHECK-NOT: @llvm.x86.avx2.paddus.w
  // CHECK: call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_adds_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_adds_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_adds_epu16
  // CHECK-NOT: @llvm.x86.avx2.paddus.w
  // CHECK: call <16 x i16> @llvm.uadd.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_adds_epu16(__U,__A,__B); 
}
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
__m128i test_mm_maskz_subs_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epi8
  // CHECK: @llvm.ssub.sat.v16i8
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_subs_epi8(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epi8
  // CHECK: @llvm.ssub.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_subs_epi8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epi8
  // CHECK: @llvm.ssub.sat.v32i8
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_subs_epi8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epi16
  // CHECK: @llvm.ssub.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_subs_epi16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epi16
  // CHECK: @llvm.ssub.sat.v8i16
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_subs_epi16(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epi16
  // CHECK: @llvm.ssub.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_subs_epi16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epi16
  // CHECK: @llvm.ssub.sat.v16i16
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_subs_epi16(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epu8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epu8
  // CHECK-NOT: @llvm.x86.sse2.psubus.b
  // CHECK: call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_subs_epu8(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epu8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epu8
  // CHECK-NOT: @llvm.x86.sse2.psubus.b
  // CHECK: call <16 x i8> @llvm.usub.sat.v16i8(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_subs_epu8(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epu8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epu8
  // CHECK-NOT: @llvm.x86.avx2.psubus.b
  // CHECK: call <32 x i8> @llvm.usub.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_subs_epu8(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epu8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epu8
  // CHECK-NOT: @llvm.x86.avx2.psubus.b
  // CHECK: call <32 x i8> @llvm.usub.sat.v32i8(<32 x i8> %{{.*}}, <32 x i8> %{{.*}})
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_subs_epu8(__U,__A,__B); 
}
__m128i test_mm_mask_subs_epu16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_subs_epu16
  // CHECK-NOT: @llvm.x86.sse2.psubus.w
  // CHECK: call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_subs_epu16(__W,__U,__A,__B); 
}
__m128i test_mm_maskz_subs_epu16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_subs_epu16
  // CHECK-NOT: @llvm.x86.sse2.psubus.w
  // CHECK: call <8 x i16> @llvm.usub.sat.v8i16(<8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_subs_epu16(__U,__A,__B); 
}
__m256i test_mm256_mask_subs_epu16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_subs_epu16
  // CHECK-NOT: @llvm.x86.avx2.psubus.w
  // CHECK: call <16 x i16> @llvm.usub.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_subs_epu16(__W,__U,__A,__B); 
}
__m256i test_mm256_maskz_subs_epu16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_subs_epu16
  // CHECK-NOT: @llvm.x86.avx2.psubus.w
  // CHECK: call <16 x i16> @llvm.usub.sat.v16i16(<16 x i16> %{{.*}}, <16 x i16> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_subs_epu16(__U,__A,__B); 
}


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

__m128i test_mm_maskz_mulhrs_epi16(__mmask8 __U, __m128i __X, __m128i __Y) {
  // CHECK-LABEL: test_mm_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.ssse3.pmul.hr.sw
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_mulhrs_epi16(__U, __X, __Y); 
}

__m256i test_mm256_mask_mulhrs_epi16(__m256i __W, __mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_mask_mulhrs_epi16
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_mulhrs_epi16(__W, __U, __X, __Y); 
}

__m256i test_mm256_maskz_mulhrs_epi16(__mmask16 __U, __m256i __X, __m256i __Y) {
  // CHECK-LABEL: test_mm256_maskz_mulhrs_epi16
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_mulhrs_epi16(__U, __X, __Y); 
}

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

__m128i test_mm_maskz_unpackhi_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpackhi_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpackhi_epi8(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpackhi_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpackhi_epi8(__U, __A, __B); 
}

__m128i test_mm_mask_unpackhi_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpackhi_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpackhi_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_unpackhi_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_unpackhi_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpackhi_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpackhi_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_unpackhi_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpacklo_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_unpacklo_epi8(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpacklo_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_unpacklo_epi8(__U, __A, __B); 
}

__m128i test_mm_mask_unpacklo_epi16(__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m128i test_mm_maskz_unpacklo_epi16(__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_unpacklo_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m256i test_mm256_mask_unpacklo_epi16(__m256i __W, __mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_unpacklo_epi16(__W, __U, __A, __B); 
}

__m256i test_mm256_maskz_unpacklo_epi16(__mmask16 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_unpacklo_epi16
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_unpacklo_epi16(__U, __A, __B); 
}

__m128i test_mm_mask_cvtepi8_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtepi8_epi16
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepi8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtepi8_epi16
  // CHECK: sext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_cvtepi8_epi16(__U, __A); 
}

__m256i test_mm256_mask_cvtepi8_epi16(__m256i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_cvtepi8_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepi8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtepi8_epi16
  // CHECK: sext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_cvtepi8_epi16(__U, __A); 
}

__m128i test_mm_mask_cvtepu8_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_cvtepu8_epi16
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m128i test_mm_maskz_cvtepu8_epi16(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_cvtepu8_epi16
  // CHECK: zext <8 x i8> %{{.*}} to <8 x i16>
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_cvtepu8_epi16(__U, __A); 
}

__m256i test_mm256_mask_cvtepu8_epi16(__m256i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_cvtepu8_epi16(__W, __U, __A); 
}

__m256i test_mm256_maskz_cvtepu8_epi16(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_cvtepu8_epi16
  // CHECK: zext <16 x i8> %{{.*}} to <16 x i16>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_cvtepu8_epi16(__U, __A); 
}

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

__m128i test_mm_mask_slli_epi16(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_slli_epi16
  // CHECK: @llvm.x86.sse2.pslli.w
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_slli_epi16(__W, __U, __A, 5); 
}

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
  // CHECK: @llvm.masked.load.v8i16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_mask_loadu_epi16(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi16(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: test_mm_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v8i16.p0(ptr %{{.*}}, i32 1, <8 x i1> %{{.*}}, <8 x i16> %{{.*}})
  return _mm_maskz_loadu_epi16(__U, __P); 
}

__m256i test_mm256_loadu_epi16(void const *__P) {
  // CHECK-LABEL: test_mm256_loadu_epi16
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm256_loadu_epi16(__P);
}

__m256i test_mm256_mask_loadu_epi16(__m256i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_mask_loadu_epi16
  // CHECK: @llvm.masked.load.v16i16.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_mask_loadu_epi16(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi16(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_maskz_loadu_epi16
  // CHECK: @llvm.masked.load.v16i16.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i16> %{{.*}})
  return _mm256_maskz_loadu_epi16(__U, __P); 
}

__m128i test_mm_loadu_epi8(void const *__P) {
  // CHECK-LABEL: test_mm_loadu_epi8
  // CHECK: load <2 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm_loadu_epi8(__P);
}

__m128i test_mm_mask_loadu_epi8(__m128i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v16i8.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_mask_loadu_epi8(__W, __U, __P); 
}

__m128i test_mm_maskz_loadu_epi8(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: test_mm_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v16i8.p0(ptr %{{.*}}, i32 1, <16 x i1> %{{.*}}, <16 x i8> %{{.*}})
  return _mm_maskz_loadu_epi8(__U, __P); 
}

__m256i test_mm256_loadu_epi8(void const *__P) {
  // CHECK-LABEL: test_mm256_loadu_epi8
  // CHECK: load <4 x i64>, ptr %{{.*}}, align 1{{$}}
  return _mm256_loadu_epi8(__P);
}

__m256i test_mm256_mask_loadu_epi8(__m256i __W, __mmask32 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_mask_loadu_epi8
  // CHECK: @llvm.masked.load.v32i8.p0(ptr %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_mask_loadu_epi8(__W, __U, __P); 
}

__m256i test_mm256_maskz_loadu_epi8(__mmask32 __U, void const *__P) {
  // CHECK-LABEL: test_mm256_maskz_loadu_epi8
  // CHECK: @llvm.masked.load.v32i8.p0(ptr %{{.*}}, i32 1, <32 x i1> %{{.*}}, <32 x i8> %{{.*}})
  return _mm256_maskz_loadu_epi8(__U, __P); 
}

void test_mm_storeu_epi16(void *__p, __m128i __a) {
  // CHECK-LABEL: test_mm_storeu_epi16
  // check: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm_storeu_epi16(__p, __a);
}

void test_mm_mask_storeu_epi16(void *__P, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v8i16.p0(<8 x i16> %{{.*}}, ptr %{{.*}}, i32 1, <8 x i1> %{{.*}})
  return _mm_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm256_storeu_epi16(void *__P, __m256i __A) {
  // CHECK-LABEL: test_mm256_storeu_epi16
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm256_storeu_epi16(__P, __A);
}

void test_mm256_mask_storeu_epi16(void *__P, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_storeu_epi16
  // CHECK: @llvm.masked.store.v16i16.p0(<16 x i16> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm256_mask_storeu_epi16(__P, __U, __A); 
}

void test_mm_storeu_epi8(void *__p, __m128i __a) {
  // CHECK-LABEL: test_mm_storeu_epi8
  // check: store <2 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm_storeu_epi8(__p, __a);
}

void test_mm_mask_storeu_epi8(void *__P, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v16i8.p0(<16 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <16 x i1> %{{.*}})
  return _mm_mask_storeu_epi8(__P, __U, __A); 
}

void test_mm256_storeu_epi8(void *__P, __m256i __A) {
  // CHECK-LABEL: test_mm256_storeu_epi8
  // CHECK: store <4 x i64> %{{.*}}, ptr %{{.*}}, align 1{{$}}
  return _mm256_storeu_epi8(__P, __A);
}

void test_mm256_mask_storeu_epi8(void *__P, __mmask32 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_storeu_epi8
  // CHECK: @llvm.masked.store.v32i8.p0(<32 x i8> %{{.*}}, ptr %{{.*}}, i32 1, <32 x i1> %{{.*}})
  return _mm256_mask_storeu_epi8(__P, __U, __A); 
}
__mmask16 test_mm_test_epi8_mask(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_test_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  return _mm_test_epi8_mask(__A, __B); 
}

__mmask16 test_mm_mask_test_epi8_mask(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_test_epi8_mask
  // CHECK: and <2 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <16 x i8> %{{.*}}, %{{.*}}
  // CHECK: and <16 x i1> %{{.*}}, %{{.*}}
  return _mm_mask_test_epi8_mask(__U, __A, __B); 
}

__mmask32 test_mm256_test_epi8_mask(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_test_epi8_mask
  // CHECK: and <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: icmp ne <32 x i8> %{{.*}}, %{{.*}}
  return _mm256_test_epi8_mask(__A, __B); 
}

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

__mmask32 test_mm256_movepi8_mask(__m256i __A) {
  // CHECK-LABEL: test_mm256_movepi8_mask
  // CHECK: [[CMP:%.*]] = icmp slt <32 x i8> %{{.*}}, zeroinitializer
  return _mm256_movepi8_mask(__A); 
}

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

__m128i test_mm_maskz_broadcastb_epi8(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_broadcastb_epi8(__M, __A);
}

__m256i test_mm256_mask_broadcastb_epi8(__m256i __O, __mmask32 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_broadcastb_epi8(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastb_epi8(__mmask32 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_broadcastb_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_broadcastb_epi8(__M, __A);
}

__m128i test_mm_mask_broadcastw_epi16(__m128i __O, __mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_mask_broadcastw_epi16(__O, __M, __A);
}

__m128i test_mm_maskz_broadcastw_epi16(__mmask8 __M, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK: select <8 x i1> %{{.*}}, <8 x i16> %{{.*}}, <8 x i16> %{{.*}}
  return _mm_maskz_broadcastw_epi16(__M, __A);
}

__m256i test_mm256_mask_broadcastw_epi16(__m256i __O, __mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_mask_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_mask_broadcastw_epi16(__O, __M, __A);
}

__m256i test_mm256_maskz_broadcastw_epi16(__mmask16 __M, __m128i __A) {
  // CHECK-LABEL: test_mm256_maskz_broadcastw_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK: select <16 x i1> %{{.*}}, <16 x i16> %{{.*}}, <16 x i16> %{{.*}}
  return _mm256_maskz_broadcastw_epi16(__M, __A);
}
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
__m128i test_mm_permutexvar_epi16(__m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.128
  return _mm_permutexvar_epi16(__A, __B); 
}

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

__m256i test_mm256_permutexvar_epi16(__m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_permutexvar_epi16
  // CHECK: @llvm.x86.avx512.permvar.hi.256
  return _mm256_permutexvar_epi16(__A, __B); 
}

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
__m128i test_mm_mask_alignr_epi8(__m128i __W, __mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_mask_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}

__m128i test_mm_maskz_alignr_epi8(__mmask16 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: test_mm_maskz_alignr_epi8
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17>
  // CHECK: select <16 x i1> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}
  return _mm_maskz_alignr_epi8(__U, __A, __B, 2); 
}

__m256i test_mm256_mask_alignr_epi8(__m256i __W, __mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_mask_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_mask_alignr_epi8(__W, __U, __A, __B, 2); 
}

__m256i test_mm256_maskz_alignr_epi8(__mmask32 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: test_mm256_maskz_alignr_epi8
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK: select <32 x i1> %{{.*}}, <32 x i8> %{{.*}}, <32 x i8> %{{.*}}
  return _mm256_maskz_alignr_epi8(__U, __A, __B, 2); 
}

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

__mmask16 test_mm256_movepi16_mask(__m256i __A) {
  // CHECK-LABEL: test_mm256_movepi16_mask
  // CHECK: [[CMP:%.*]] = icmp slt <16 x i16> %{{.*}}, zeroinitializer
  return _mm256_movepi16_mask(__A); 
}

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
