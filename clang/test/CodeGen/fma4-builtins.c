// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +fma4 -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +fma4 -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128 test_mm_macc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmadd.ps
  // CHECK-ASM: vfmaddps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macc_ps(a, b, c);
}

__m128d test_mm_macc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmadd.pd
  // CHECK-ASM: vfmaddpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macc_pd(a, b, c);
}

__m128 test_mm_macc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmadd.ss
  // CHECK-ASM: vfmaddss %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macc_ss(a, b, c);
}

__m128d test_mm_macc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmadd.sd
  // CHECK-ASM: vfmaddsd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_macc_sd(a, b, c);
}

__m128 test_mm_msub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmsub.ps
  // CHECK-ASM: vfmsubps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_msub_ps(a, b, c);
}

__m128d test_mm_msub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmsub.pd
  // CHECK-ASM: vfmsubpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_msub_pd(a, b, c);
}

__m128 test_mm_msub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmsub.ss
  // CHECK-ASM: vfmsubss %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_msub_ss(a, b, c);
}

__m128d test_mm_msub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmsub.sd
  // CHECK-ASM: vfmsubsd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_msub_sd(a, b, c);
}

__m128 test_mm_nmacc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmadd.ps
  // CHECK-ASM: vfnmaddps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmacc_ps(a, b, c);
}

__m128d test_mm_nmacc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmadd.pd
  // CHECK-ASM: vfnmaddpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmacc_pd(a, b, c);
}

__m128 test_mm_nmacc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmadd.ss
  // CHECK-ASM: vfnmaddss %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmacc_ss(a, b, c);
}

__m128d test_mm_nmacc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmadd.sd
  // CHECK-ASM: vfnmaddsd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmacc_sd(a, b, c);
}

__m128 test_mm_nmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmsub.ps
  // CHECK-ASM: vfnmsubps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmsub_ps(a, b, c);
}

__m128d test_mm_nmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmsub.pd
  // CHECK-ASM: vfnmsubpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmsub_pd(a, b, c);
}

__m128 test_mm_nmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfnmsub.ss
  // CHECK-ASM: vfnmsubss %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmsub_ss(a, b, c);
}

__m128d test_mm_nmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfnmsub.sd
  // CHECK-ASM: vfnmsubsd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_nmsub_sd(a, b, c);
}

__m128 test_mm_maddsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.ps
  // CHECK-ASM: vfmaddsubps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maddsub_ps(a, b, c);
}

__m128d test_mm_maddsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.pd
  // CHECK-ASM: vfmaddsubpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_maddsub_pd(a, b, c);
}

__m128 test_mm_msubadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.ps
  // CHECK-ASM: vfmsubaddps %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_msubadd_ps(a, b, c);
}

__m128d test_mm_msubadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.pd
  // CHECK-ASM: vfmsubaddpd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_msubadd_pd(a, b, c);
}

__m256 test_mm256_macc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmadd.ps.256
  // CHECK-ASM: vfmaddps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_macc_ps(a, b, c);
}

__m256d test_mm256_macc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmadd.pd.256
  // CHECK-ASM: vfmaddpd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_macc_pd(a, b, c);
}

__m256 test_mm256_msub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmsub.ps.256
  // CHECK-ASM: vfmsubps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_msub_ps(a, b, c);
}

__m256d test_mm256_msub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmsub.pd.256
  // CHECK-ASM: vfmsubpd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_msub_pd(a, b, c);
}

__m256 test_mm256_nmacc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfnmadd.ps.256
  // CHECK-ASM: vfnmaddps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_nmacc_ps(a, b, c);
}

__m256d test_mm256_nmacc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfnmadd.pd.256
  // CHECK-ASM: vfnmaddpd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_nmacc_pd(a, b, c);
}

__m256 test_mm256_nmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfnmsub.ps.256
  // CHECK-ASM: vfnmsubps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_nmsub_ps(a, b, c);
}

__m256d test_mm256_nmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfnmsub.pd.256
  // CHECK-ASM: vfnmsubpd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_nmsub_pd(a, b, c);
}

__m256 test_mm256_maddsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.ps.256
  // CHECK-ASM: vfmaddsubps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_maddsub_ps(a, b, c);
}

__m256d test_mm256_maddsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmaddsub.pd.256
  // CHECK-ASM: vfmaddsubpd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_maddsub_pd(a, b, c);
}

__m256 test_mm256_msubadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.ps.256
  // CHECK-ASM: vfmsubaddps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_msubadd_ps(a, b, c);
}

__m256d test_mm256_msubadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK: @llvm.x86.fma.vfmsubadd.pd.256
  // CHECK-ASM: vfmsubaddpd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_msubadd_pd(a, b, c);
}
