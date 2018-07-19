// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +fma4 -emit-llvm -o - -Wall -Werror | FileCheck %s


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
  // CHECK: @llvm.x86.fma4.vfmadd.ss
  return _mm_macc_ss(a, b, c);
}

__m128d test_mm_macc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_macc_sd
  // CHECK: @llvm.x86.fma4.vfmadd.sd
  return _mm_macc_sd(a, b, c);
}

__m128 test_mm_msub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msub_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.ps(<4 x float> %{{.+}}, <4 x float> %{{.+}}, <4 x float> [[NEG]])
  return _mm_msub_ps(a, b, c);
}

__m128d test_mm_msub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msub_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.pd(<2 x double> %{{.+}}, <2 x double> %{{.+}}, <2 x double> [[NEG]])
  return _mm_msub_pd(a, b, c);
}

__m128 test_mm_msub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msub_ss
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma4.vfmadd.ss(<4 x float> %{{.+}}, <4 x float> %{{.+}}, <4 x float> [[NEG]])
  return _mm_msub_ss(a, b, c);
}

__m128d test_mm_msub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msub_sd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma4.vfmadd.sd(<2 x double> %{{.+}}, <2 x double> %{{.+}}, <2 x double> [[NEG]])
  return _mm_msub_sd(a, b, c);
}

__m128 test_mm_nmacc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmacc_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.ps(<4 x float> [[NEG]], <4 x float> %{{.+}}, <4 x float> %{{.+}})
  return _mm_nmacc_ps(a, b, c);
}

__m128d test_mm_nmacc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmacc_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.pd(<2 x double> [[NEG]], <2 x double> %{{.+}}, <2 x double> %{{.+}})
  return _mm_nmacc_pd(a, b, c);
}

__m128 test_mm_nmacc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmacc_ss
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma4.vfmadd.ss(<4 x float> [[NEG]], <4 x float> %{{.+}}, <4 x float> %{{.+}})
  return _mm_nmacc_ss(a, b, c);
}

__m128d test_mm_nmacc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmacc_sd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma4.vfmadd.sd(<2 x double> [[NEG]], <2 x double> %{{.+}}, <2 x double> %{{.+}})
  return _mm_nmacc_sd(a, b, c);
}

__m128 test_mm_nmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmsub_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.ps(<4 x float> [[NEG]], <4 x float> %{{.+}}, <4 x float> [[NEG2]])
  return _mm_nmsub_ps(a, b, c);
}

__m128d test_mm_nmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmsub_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.pd(<2 x double> [[NEG]], <2 x double> %{{.+}}, <2 x double> [[NEG2]])
  return _mm_nmsub_pd(a, b, c);
}

__m128 test_mm_nmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmsub_ss
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma4.vfmadd.ss(<4 x float> [[NEG]], <4 x float> %{{.+}}, <4 x float> [[NEG2]])
  return _mm_nmsub_ss(a, b, c);
}

__m128d test_mm_nmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmsub_sd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma4.vfmadd.sd(<2 x double> [[NEG]], <2 x double> %{{.+}}, <2 x double> [[NEG2]])
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
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.+}}, <4 x float> %{{.+}}, <4 x float> [[NEG]])
  return _mm_msubadd_ps(a, b, c);
}

__m128d test_mm_msubadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msubadd_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.+}}, <2 x double> %{{.+}}, <2 x double> [[NEG]])
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
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: @llvm.x86.fma.vfmadd.ps.256(<8 x float> %{{.+}}, <8 x float> %{{.+}}, <8 x float> [[NEG]])
  return _mm256_msub_ps(a, b, c);
}

__m256d test_mm256_msub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_msub_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.pd.256(<4 x double> %{{.+}}, <4 x double> %{{.+}}, <4 x double> [[NEG]])
  return _mm256_msub_pd(a, b, c);
}

__m256 test_mm256_nmacc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_nmacc_ps
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: @llvm.x86.fma.vfmadd.ps.256(<8 x float> [[NEG]], <8 x float> %{{.+}}, <8 x float> %{{.+}})
  return _mm256_nmacc_ps(a, b, c);
}

__m256d test_mm256_nmacc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_nmacc_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.pd.256(<4 x double> [[NEG]], <4 x double> %{{.+}}, <4 x double> %{{.+}})
  return _mm256_nmacc_pd(a, b, c);
}

__m256 test_mm256_nmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_nmsub_ps
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: [[NEG2:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: @llvm.x86.fma.vfmadd.ps.256(<8 x float> [[NEG]], <8 x float> %{{.+}}, <8 x float> [[NEG2]])
  return _mm256_nmsub_ps(a, b, c);
}

__m256d test_mm256_nmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_nmsub_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmadd.pd.256(<4 x double> [[NEG]], <4 x double> %{{.+}}, <4 x double> [[NEG2]])
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
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.+}}, <8 x float> [[NEG]])
  return _mm256_msubadd_ps(a, b, c);
}

__m256d test_mm256_msubadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_msubadd_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.+}}, <4 x double> %{{.+}}, <4 x double> [[NEG]])
  return _mm256_msubadd_pd(a, b, c);
}
