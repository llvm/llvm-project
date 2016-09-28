// RUN: %clang_cc1 -ffreestanding -O3 -triple x86_64-apple-macosx10.8.0 -target-feature +sse4.1 -emit-llvm %s -o - | FileCheck %s
// FIXME: This test currently depends on optimization - it should be rewritten to avoid it.


#include <emmintrin.h>

// Byte-shifts look reversed due to xmm register layout
__m128 test_mm_slli_si128(__m128 a) {
  // CHECK-LABEL: @test_mm_slli_si128
  // CHECK: shufflevector <16 x i8> <{{.*}}, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8> {{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  return _mm_slli_si128(a, 5);
}

__m128 test_mm_slli_si128_0(__m128 a) {
  // CHECK-LABEL: @test_mm_slli_si128_0
  // CHECK-NOT: shufflevector
  return _mm_slli_si128(a, 0);
}

__m128 test_mm_slli_si128_16(__m128 a) {
  // CHECK-LABEL: @test_mm_slli_si128_16
  // CHECK-NOT: shufflevector
  return _mm_slli_si128(a, 16);
}

__m128 test_mm_srli_si128(__m128 a) {
  // CHECK-LABEL: @test_mm_srli_si128
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, {{.*}}>, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  return _mm_srli_si128(a, 5);
}

__m128 test_mm_srli_si128_0(__m128 a) {
  // CHECK-LABEL: @test_mm_srli_si128_0
  // CHECK-NOT: shufflevector
  return _mm_srli_si128(a, 0);
}

__m128 test_mm_srli_si128_16(__m128 a) {
  // CHECK-LABEL: @test_mm_srli_si128_16
  // CHECK-NOT: shufflevector
  return _mm_srli_si128(a, 16);
}
