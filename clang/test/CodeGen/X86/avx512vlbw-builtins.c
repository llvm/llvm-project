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

TEST_CONSTEXPR(_mm_cmpeq_epi8_mask(
    ((__m128i)(__v16qi){5, 3, 7, 2, 9, 3, 7, 1, 5, 4, 8, 2, 9, 6, 7, 5}),
    ((__m128i)(__v16qi){5, 2, 7, 3, 9, 4, 6, 1, 5, 3, 8, 1, 9, 5, 7, 5})
) == (__mmask16)0xd595);

TEST_CONSTEXPR(_mm_cmplt_epi8_mask(
    ((__m128i)(__v16qi){1, 5, 3, 7, 2, 8, 4, 6, 9, 5, 3, 11, 2, 6, 15, 8}),
    ((__m128i)(__v16qi){2, 4, 6, 8, 3, 5, 7, 9, 4, 6, 8, 10, 5, 7, 9, 11})
) == (__mmask16)0xb6dd);

TEST_CONSTEXPR(_mm_cmple_epi8_mask(
    ((__m128i)(__v16qi){1, 3, 5, 7, 2, 6, 6, 8, 1, 3, 9, 7, 2, 4, 6, 10}),
    ((__m128i)(__v16qi){2, 3, 4, 7, 3, 4, 5, 8, 2, 3, 4, 7, 3, 4, 5, 8})
) == (__mmask16)0x3b9b);

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
