// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +fma4 -emit-llvm -o - -Wall -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128 test_mm_macc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_macc_ps
  // CHECK: @llvm.x86.fma.vfmadd.ps
  return _mm_macc_ps(a, b, c);
}

__m128d test_mm_macc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_macc_pd
  // CHECK: @llvm.x86.fma.vfmadd.pd
  return _mm_macc_pd(a, b, c);
}

__m128 test_mm_macc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_macc_ss
  // CHECK: @llvm.x86.fma.vfmadd.ss
  return _mm_macc_ss(a, b, c);
}

__m128d test_mm_macc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_macc_sd
  // CHECK: @llvm.x86.fma.vfmadd.sd
  return _mm_macc_sd(a, b, c);
}

__m128 test_mm_msub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msub_ps
  // CHECK: @llvm.x86.fma.vfmsub.ps
  return _mm_msub_ps(a, b, c);
}

__m128d test_mm_msub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msub_pd
  // CHECK: @llvm.x86.fma.vfmsub.pd
  return _mm_msub_pd(a, b, c);
}

__m128 test_mm_msub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msub_ss
  // CHECK: @llvm.x86.fma.vfmsub.ss
  return _mm_msub_ss(a, b, c);
}

__m128d test_mm_msub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msub_sd
  // CHECK: @llvm.x86.fma.vfmsub.sd
  return _mm_msub_sd(a, b, c);
}

__m128 test_mm_nmacc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmacc_ps
  // CHECK: @llvm.x86.fma.vfnmadd.ps
  return _mm_nmacc_ps(a, b, c);
}

__m128d test_mm_nmacc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmacc_pd
  // CHECK: @llvm.x86.fma.vfnmadd.pd
  return _mm_nmacc_pd(a, b, c);
}

__m128 test_mm_nmacc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmacc_ss
  // CHECK: @llvm.x86.fma.vfnmadd.ss
  return _mm_nmacc_ss(a, b, c);
}

__m128d test_mm_nmacc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmacc_sd
  // CHECK: @llvm.x86.fma.vfnmadd.sd
  return _mm_nmacc_sd(a, b, c);
}

__m128 test_mm_nmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmsub_ps
  // CHECK: @llvm.x86.fma.vfnmsub.ps
  return _mm_nmsub_ps(a, b, c);
}

__m128d test_mm_nmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmsub_pd
  // CHECK: @llvm.x86.fma.vfnmsub.pd
  return _mm_nmsub_pd(a, b, c);
}

__m128 test_mm_nmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmsub_ss
  // CHECK: @llvm.x86.fma.vfnmsub.ss
  return _mm_nmsub_ss(a, b, c);
}

__m128d test_mm_nmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmsub_sd
  // CHECK: @llvm.x86.fma.vfnmsub.sd
  return _mm_nmsub_sd(a, b, c);
}

__m128 test_mm_maddsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_maddsub_ps
  // CHECK: @llvm.x86.fma.vfmaddsub.ps
  return _mm_maddsub_ps(a, b, c);
}

__m128d test_mm_maddsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_maddsub_pd
  // CHECK: @llvm.x86.fma.vfmaddsub.pd
  return _mm_maddsub_pd(a, b, c);
}

__m128 test_mm_msubadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msubadd_ps
  // CHECK: @llvm.x86.fma.vfmsubadd.ps
  return _mm_msubadd_ps(a, b, c);
}

__m128d test_mm_msubadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msubadd_pd
  // CHECK: @llvm.x86.fma.vfmsubadd.pd
  return _mm_msubadd_pd(a, b, c);
}

__m256 test_mm256_macc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_macc_ps
  // CHECK: @llvm.x86.fma.vfmadd.ps.256
  return _mm256_macc_ps(a, b, c);
}

__m256d test_mm256_macc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_macc_pd
  // CHECK: @llvm.x86.fma.vfmadd.pd.256
  return _mm256_macc_pd(a, b, c);
}

__m256 test_mm256_msub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_msub_ps
  // CHECK: @llvm.x86.fma.vfmsub.ps.256
  return _mm256_msub_ps(a, b, c);
}

__m256d test_mm256_msub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_msub_pd
  // CHECK: @llvm.x86.fma.vfmsub.pd.256
  return _mm256_msub_pd(a, b, c);
}

__m256 test_mm256_nmacc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_nmacc_ps
  // CHECK: @llvm.x86.fma.vfnmadd.ps.256
  return _mm256_nmacc_ps(a, b, c);
}

__m256d test_mm256_nmacc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_nmacc_pd
  // CHECK: @llvm.x86.fma.vfnmadd.pd.256
  return _mm256_nmacc_pd(a, b, c);
}

__m256 test_mm256_nmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_nmsub_ps
  // CHECK: @llvm.x86.fma.vfnmsub.ps.256
  return _mm256_nmsub_ps(a, b, c);
}

__m256d test_mm256_nmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_nmsub_pd
  // CHECK: @llvm.x86.fma.vfnmsub.pd.256
  return _mm256_nmsub_pd(a, b, c);
}

__m256 test_mm256_maddsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_maddsub_ps
  // CHECK: @llvm.x86.fma.vfmaddsub.ps.256
  return _mm256_maddsub_ps(a, b, c);
}

__m256d test_mm256_maddsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_maddsub_pd
  // CHECK: @llvm.x86.fma.vfmaddsub.pd.256
  return _mm256_maddsub_pd(a, b, c);
}

__m256 test_mm256_msubadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_msubadd_ps
  // CHECK: @llvm.x86.fma.vfmsubadd.ps.256
  return _mm256_msubadd_ps(a, b, c);
}

__m256d test_mm256_msubadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_msubadd_pd
  // CHECK: @llvm.x86.fma.vfmsubadd.pd.256
  return _mm256_msubadd_pd(a, b, c);
}
