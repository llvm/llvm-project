// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx2 -target-feature +avxneconvert \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown  -target-feature +avx2 -target-feature +avxneconvert \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

__m128 test_mm_bcstnebf16_ps(const void *__A) {
  // CHECK-LABEL: @test_mm_bcstnebf16_ps(
  // CHECK: call <4 x float> @llvm.x86.vbcstnebf162ps128(ptr %{{.*}})
  return _mm_bcstnebf16_ps(__A);
}

__m256 test_mm256_bcstnebf16_ps(const void *__A) {
  // CHECK-LABEL: @test_mm256_bcstnebf16_ps(
  // CHECK: call <8 x float> @llvm.x86.vbcstnebf162ps256(ptr %{{.*}})
  return _mm256_bcstnebf16_ps(__A);
}

__m128 test_mm_bcstnesh_ps(const void *__A) {
  // CHECK-LABEL: @test_mm_bcstnesh_ps(
  // CHECK: call <4 x float> @llvm.x86.vbcstnesh2ps128(ptr %{{.*}})
  return _mm_bcstnesh_ps(__A);
}

__m256 test_mm256_bcstnesh_ps(const void *__A) {
  // CHECK-LABEL: @test_mm256_bcstnesh_ps(
  // CHECK: call <8 x float> @llvm.x86.vbcstnesh2ps256(ptr %{{.*}})
  return _mm256_bcstnesh_ps(__A);
}

__m128 test_mm_cvtneebf16_ps(const __m128bh *__A) {
  // CHECK-LABEL: @test_mm_cvtneebf16_ps(
  // CHECK: call <4 x float> @llvm.x86.vcvtneebf162ps128(ptr %{{.*}})
  return _mm_cvtneebf16_ps(__A);
}

__m256 test_mm256_cvtneebf16_ps(const __m256bh *__A) {
  // CHECK-LABEL: @test_mm256_cvtneebf16_ps(
  // CHECK: call <8 x float> @llvm.x86.vcvtneebf162ps256(ptr %{{.*}})
  return _mm256_cvtneebf16_ps(__A);
}

__m128 test_mm_cvtneeph_ps(const __m128h *__A) {
  // CHECK-LABEL: @test_mm_cvtneeph_ps(
  // CHECK: call <4 x float> @llvm.x86.vcvtneeph2ps128(ptr %{{.*}})
  return _mm_cvtneeph_ps(__A);
}

__m256 test_mm256_cvtneeph_ps(const __m256h *__A) {
  // CHECK-LABEL: @test_mm256_cvtneeph_ps(
  // CHECK: call <8 x float> @llvm.x86.vcvtneeph2ps256(ptr %{{.*}})
  return _mm256_cvtneeph_ps(__A);
}

__m128 test_mm_cvtneobf16_ps(const __m128bh *__A) {
  // CHECK-LABEL: @test_mm_cvtneobf16_ps(
  // CHECK: call <4 x float> @llvm.x86.vcvtneobf162ps128(ptr %{{.*}})
  return _mm_cvtneobf16_ps(__A);
}

__m256 test_mm256_cvtneobf16_ps(const __m256bh *__A) {
  // CHECK-LABEL: @test_mm256_cvtneobf16_ps(
  // CHECK: call <8 x float> @llvm.x86.vcvtneobf162ps256(ptr %{{.*}})
  return _mm256_cvtneobf16_ps(__A);
}

__m128 test_mm_cvtneoph_ps(const __m128h *__A) {
  // CHECK-LABEL: @test_mm_cvtneoph_ps(
  // CHECK: call <4 x float> @llvm.x86.vcvtneoph2ps128(ptr %{{.*}})
  return _mm_cvtneoph_ps(__A);
}

__m256 test_mm256_cvtneoph_ps(const __m256h *__A) {
  // CHECK-LABEL: @test_mm256_cvtneoph_ps(
  // CHECK: call <8 x float> @llvm.x86.vcvtneoph2ps256(ptr %{{.*}})
  return _mm256_cvtneoph_ps(__A);
}

__m128bh test_mm_cvtneps_avx_pbh(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtneps_avx_pbh(
  // CHECK: call <8 x bfloat> @llvm.x86.vcvtneps2bf16128(<4 x float> %{{.*}})
  return _mm_cvtneps_avx_pbh(__A);
}

__m128bh test_mm256_cvtneps_avx_pbh(__m256 __A) {
  // CHECK-LABEL: @test_mm256_cvtneps_avx_pbh(
  // CHECK: call <8 x bfloat> @llvm.x86.vcvtneps2bf16256(<8 x float> %{{.*}})
  return _mm256_cvtneps_avx_pbh(__A);
}
