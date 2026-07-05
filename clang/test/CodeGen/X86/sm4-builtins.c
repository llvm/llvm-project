// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +sm4 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +sm4 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m128i test_mm_sm4key4_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_sm4key4_epi32(
  // CHECK: call <4 x i32> @llvm.x86.vsm4key4128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sm4key4_epi32(__A, __B);
}

__m256i test_mm256_sm4key4_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_sm4key4_epi32(
  // CHECK: call <8 x i32> @llvm.x86.vsm4key4256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_sm4key4_epi32(__A, __B);
}

__m128i test_mm_sm4rnds4_epi32(__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_sm4rnds4_epi32(
  // CHECK: call <4 x i32> @llvm.x86.vsm4rnds4128(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  return _mm_sm4rnds4_epi32(__A, __B);
}

__m256i test_mm256_sm4rnds4_epi32(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_sm4rnds4_epi32(
  // CHECK: call <8 x i32> @llvm.x86.vsm4rnds4256(<8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return _mm256_sm4rnds4_epi32(__A, __B);
}
