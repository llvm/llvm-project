// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

__m128i test_mm_move_epi32(__m128i A) {
  // CHECK-LABEL: test_mm_move_epi32
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 4, i32 4>
  return _mm_move_epi32(A);
}

__m128i test_mm_move_epi16(__m128i A) {
  // CHECK-LABEL: test_mm_move_epi16
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  return _mm_move_epi16(A);
}
