// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +fma -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

__m128 test_mm_fmadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmadd.ps
  return _mm_fmadd_ps(a, b, c);
}

__m128d test_mm_fmadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmadd.pd
  return _mm_fmadd_pd(a, b, c);
}

__m128 test_mm_fmadd_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmadd.ss
  return _mm_fmadd_ss(a, b, c);
}

__m128d test_mm_fmadd_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmadd.sd
  return _mm_fmadd_sd(a, b, c);
}

__m128 test_mm_fmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmsub.ps
  return _mm_fmsub_ps(a, b, c);
}

__m128d test_mm_fmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmsub.pd
  return _mm_fmsub_pd(a, b, c);
}

__m128 test_mm_fmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmsub.ss
  return _mm_fmsub_ss(a, b, c);
}

__m128d test_mm_fmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmsub.sd
  return _mm_fmsub_sd(a, b, c);
}

__m128 test_mm_fnmadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmadd.ps
  return _mm_fnmadd_ps(a, b, c);
}

__m128d test_mm_fnmadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmadd.pd
  return _mm_fnmadd_pd(a, b, c);
}

__m128 test_mm_fnmadd_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmadd.ss
  return _mm_fnmadd_ss(a, b, c);
}

__m128d test_mm_fnmadd_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmadd.sd
  return _mm_fnmadd_sd(a, b, c);
}

__m128 test_mm_fnmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmsub.ps
  return _mm_fnmsub_ps(a, b, c);
}

__m128d test_mm_fnmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmsub.pd
  return _mm_fnmsub_pd(a, b, c);
}

__m128 test_mm_fnmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmsub.ss
  return _mm_fnmsub_ss(a, b, c);
}

__m128d test_mm_fnmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmsub.sd
  return _mm_fnmsub_sd(a, b, c);
}

__m128 test_mm_fmaddsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.ps
  return _mm_fmaddsub_ps(a, b, c);
}

__m128d test_mm_fmaddsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.pd
  return _mm_fmaddsub_pd(a, b, c);
}

__m128 test_mm_fmsubadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.ps
  return _mm_fmsubadd_ps(a, b, c);
}

__m128d test_mm_fmsubadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.pd
  return _mm_fmsubadd_pd(a, b, c);
}

__m256 test_mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmadd.ps.256
  return _mm256_fmadd_ps(a, b, c);
}

__m256d test_mm256_fmadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmadd.pd.256
  return _mm256_fmadd_pd(a, b, c);
}

__m256 test_mm256_fmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmsub.ps.256
  return _mm256_fmsub_ps(a, b, c);
}

__m256d test_mm256_fmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmsub.pd.256
  return _mm256_fmsub_pd(a, b, c);
}

__m256 test_mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfnmadd.ps.256
  return _mm256_fnmadd_ps(a, b, c);
}

__m256d test_mm256_fnmadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfnmadd.pd.256
  return _mm256_fnmadd_pd(a, b, c);
}

__m256 test_mm256_fnmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfnmsub.ps.256
  return _mm256_fnmsub_ps(a, b, c);
}

__m256d test_mm256_fnmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfnmsub.pd.256
  return _mm256_fnmsub_pd(a, b, c);
}

__m256 test_mm256_fmaddsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.ps.256
  return _mm256_fmaddsub_ps(a, b, c);
}

__m256d test_mm256_fmaddsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.pd.256
  return _mm256_fmaddsub_pd(a, b, c);
}

__m256 test_mm256_fmsubadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.ps.256
  return _mm256_fmsubadd_ps(a, b, c);
}

__m256d test_mm256_fmsubadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.pd.256
  return _mm256_fmsubadd_pd(a, b, c);
}
