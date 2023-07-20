// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +sha512 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +sha512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

__m256i test_mm256_sha512msg1_epi64(__m256i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm256_sha512msg1_epi64(
  // CHECK: call <4 x i64> @llvm.x86.vsha512msg1(<4 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm256_sha512msg1_epi64(__A, __B);
}

__m256i test_mm256_sha512msg2_epi64(__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_sha512msg2_epi64(
  // CHECK: call <4 x i64> @llvm.x86.vsha512msg2(<4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_sha512msg2_epi64(__A, __B);
}

__m256i test_mm256_sha512rnds2_epi64(__m256i __A, __m256i __B, __m128i __C) {
  // CHECK-LABEL: @test_mm256_sha512rnds2_epi64(
  // CHECK: call <4 x i64> @llvm.x86.vsha512rnds2(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm256_sha512rnds2_epi64(__A, __B, __C);
}
