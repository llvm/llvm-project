// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +sm3 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +sm3 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_sm3msg1_epi32(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_sm3msg1_epi32(
  // CHECK: call <4 x i32> @llvm.x86.vsm3msg1(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sm3msg1_epi32(__A, __B, __C);
}

__m128i test_mm_sm3msg2_epi32(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_sm3msg2_epi32(
  // CHECK: call <4 x i32> @llvm.x86.vsm3msg2(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sm3msg2_epi32(__A, __B, __C);
}

__m128i test_mm_sm3rnds2_epi32(__m128i __A, __m128i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_sm3rnds2_epi32(
  // CHECK: call <4 x i32> @llvm.x86.vsm3rnds2(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 127)
  return _mm_sm3rnds2_epi32(__A, __B, __C, 127);
}
