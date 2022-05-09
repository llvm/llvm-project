// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +fma4 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +fma4 -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

__m128 test_mm_macc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_macc_ps
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_macc_ps(a, b, c);
}

__m128d test_mm_macc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_macc_pd
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_macc_pd(a, b, c);
}

__m128 test_mm_macc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_macc_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CHECK: insertelement <4 x float> zeroinitializer, float %{{.*}}, i64 0
  return _mm_macc_ss(a, b, c);
}

__m128d test_mm_macc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_macc_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CHECK: insertelement <2 x double> zeroinitializer, double %{{.*}}, i64 0
  return _mm_macc_sd(a, b, c);
}

__m128 test_mm_msub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msub_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_msub_ps(a, b, c);
}

__m128d test_mm_msub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msub_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_msub_pd(a, b, c);
}

__m128 test_mm_msub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: [[C:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float [[C]])
  // CHECK: insertelement <4 x float> zeroinitializer, float %{{.*}}, i64 0
  return _mm_msub_ss(a, b, c);
}

__m128d test_mm_msub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: [[C:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double [[C]])
  // CHECK: insertelement <2 x double> zeroinitializer, double %{{.*}}, i64 0
  return _mm_msub_sd(a, b, c);
}

__m128 test_mm_nmacc_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmacc_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_nmacc_ps(a, b, c);
}

__m128d test_mm_nmacc_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmacc_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_nmacc_pd(a, b, c);
}

__m128 test_mm_nmacc_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmacc_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: call float @llvm.fma.f32(float [[A]], float %{{.*}}, float %{{.*}})
  // CHECK: insertelement <4 x float> zeroinitializer, float %{{.*}}, i64 0
  return _mm_nmacc_ss(a, b, c);
}

__m128d test_mm_nmacc_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmacc_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.fma.f64(double [[A]], double %{{.*}}, double %{{.*}})
  // CHECK: insertelement <2 x double> zeroinitializer, double %{{.*}}, i64 0
  return _mm_nmacc_sd(a, b, c);
}

__m128 test_mm_nmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmsub_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_nmsub_ps(a, b, c);
}

__m128d test_mm_nmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmsub_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_nmsub_pd(a, b, c);
}

__m128 test_mm_nmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_nmsub_ss
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: [[A:%.+]] = extractelement <4 x float> [[NEG]], i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: [[C:%.+]] = extractelement <4 x float> [[NEG2]], i64 0
  // CHECK: call float @llvm.fma.f32(float [[A]], float %{{.*}}, float [[C]])
  // CHECK: insertelement <4 x float> zeroinitializer, float %{{.*}}, i64 0
  return _mm_nmsub_ss(a, b, c);
}

__m128d test_mm_nmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_nmsub_sd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: [[NEG2:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: [[A:%.+]] = extractelement <2 x double> [[NEG]], i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: [[C:%.+]] = extractelement <2 x double> [[NEG2]], i64 0
  // CHECK: call double @llvm.fma.f64(double [[A]], double %{{.*}}, double [[C]])
  // CHECK: insertelement <2 x double> zeroinitializer, double %{{.*}}, i64 0
  return _mm_nmsub_sd(a, b, c);
}

__m128 test_mm_maddsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_maddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_maddsub_ps(a, b, c);
}

__m128d test_mm_maddsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_maddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_maddsub_pd(a, b, c);
}

__m128 test_mm_msubadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_msubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // CHECK: call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[NEG]])
  return _mm_msubadd_ps(a, b, c);
}

__m128d test_mm_msubadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_msubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // CHECK: call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[NEG]])
  return _mm_msubadd_pd(a, b, c);
}

__m256 test_mm256_macc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_macc_ps
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_macc_ps(a, b, c);
}

__m256d test_mm256_macc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_macc_pd
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_macc_pd(a, b, c);
}

__m256 test_mm256_msub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_msub_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_msub_ps(a, b, c);
}

__m256d test_mm256_msub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_msub_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_msub_pd(a, b, c);
}

__m256 test_mm256_nmacc_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_nmacc_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_nmacc_ps(a, b, c);
}

__m256d test_mm256_nmacc_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_nmacc_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_nmacc_pd(a, b, c);
}

__m256 test_mm256_nmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_nmsub_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: [[NEG2:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_nmsub_ps(a, b, c);
}

__m256d test_mm256_nmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_nmsub_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // CHECK: [[NEG2:%.+]] = fneg <4 x double> %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_nmsub_pd(a, b, c);
}

__m256 test_mm256_maddsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_maddsub_ps
  // CHECK-NOT: fneg
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_maddsub_ps(a, b, c);
}

__m256d test_mm256_maddsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_maddsub_pd
  // CHECK-NOT: fneg
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_maddsub_pd(a, b, c);
}

__m256 test_mm256_msubadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_msubadd_ps
  // CHECK: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // CHECK: call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> [[NEG]])
  return _mm256_msubadd_ps(a, b, c);
}

__m256d test_mm256_msubadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_msubadd_pd
  // CHECK: [[NEG:%.+]] = fneg <4 x double> {{.+}}
  // CHECK: call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> [[NEG]])
  return _mm256_msubadd_pd(a, b, c);
}
