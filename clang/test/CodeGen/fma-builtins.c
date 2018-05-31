// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +fma -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

__m128 test_mm_fmadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fmadd_ps
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_fmadd_ps(a, b, c);
}

__m128d test_mm_fmadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fmadd_pd
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_fmadd_pd(a, b, c);
}

__m128 test_mm_fmadd_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fmadd_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fmadd_ss(a, b, c);
}

__m128d test_mm_fmadd_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fmadd_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fmadd_sd(a, b, c);
}

__m128 test_mm_fmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fmsub_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_fmsub_ps(a, b, c);
}

__m128d test_mm_fmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fmsub_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_fmsub_pd(a, b, c);
}

__m128 test_mm_fmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fmsub_ss
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fmsub_ss(a, b, c);
}

__m128d test_mm_fmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fmsub_sd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fmsub_sd(a, b, c);
}

__m128 test_mm_fnmadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fnmadd_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_fnmadd_ps(a, b, c);
}

__m128d test_mm_fnmadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fnmadd_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_fnmadd_pd(a, b, c);
}

__m128 test_mm_fnmadd_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fnmadd_ss
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fnmadd_ss(a, b, c);
}

__m128d test_mm_fnmadd_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fnmadd_sd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fnmadd_sd(a, b, c);
}

__m128 test_mm_fnmsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fnmsub_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_fnmsub_ps(a, b, c);
}

__m128d test_mm_fnmsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fnmsub_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_fnmsub_pd(a, b, c);
}

__m128 test_mm_fnmsub_ss(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fnmsub_ss
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CHECK: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fnmsub_ss(a, b, c);
}

__m128d test_mm_fnmsub_sd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fnmsub_sd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fnmsub_sd(a, b, c);
}

__m128 test_mm_fmaddsub_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fmaddsub_ps
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_fmaddsub_ps(a, b, c);
}

__m128d test_mm_fmaddsub_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fmaddsub_pd
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: select <2 x i1> <i1 true, i1 false>, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_fmaddsub_pd(a, b, c);
}

__m128 test_mm_fmsubadd_ps(__m128 a, __m128 b, __m128 c) {
  // CHECK-LABEL: test_mm_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x float> %{{.*}}, <4 x float> %{{.*}}
  return _mm_fmsubadd_ps(a, b, c);
}

__m128d test_mm_fmsubadd_pd(__m128d a, __m128d b, __m128d c) {
  // CHECK-LABEL: test_mm_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: select <2 x i1> <i1 true, i1 false>, <2 x double> %{{.*}}, <2 x double> %{{.*}}
  return _mm_fmsubadd_pd(a, b, c);
}

__m256 test_mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_fmadd_ps
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_fmadd_ps(a, b, c);
}

__m256d test_mm256_fmadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_fmadd_pd
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_fmadd_pd(a, b, c);
}

__m256 test_mm256_fmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_fmsub_ps
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_fmsub_ps(a, b, c);
}

__m256d test_mm256_fmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_fmsub_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_fmsub_pd(a, b, c);
}

__m256 test_mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_fnmadd_ps
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_fnmadd_ps(a, b, c);
}

__m256d test_mm256_fnmadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_fnmadd_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_fnmadd_pd(a, b, c);
}

__m256 test_mm256_fnmsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_fnmsub_ps
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: [[NEG2:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return _mm256_fnmsub_ps(a, b, c);
}

__m256d test_mm256_fnmsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_fnmsub_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: [[NEG2:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return _mm256_fnmsub_pd(a, b, c);
}

__m256 test_mm256_fmaddsub_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_fmaddsub_ps
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_fmaddsub_ps(a, b, c);
}

__m256d test_mm256_fmaddsub_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_fmaddsub_pd
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_fmaddsub_pd(a, b, c);
}

__m256 test_mm256_fmsubadd_ps(__m256 a, __m256 b, __m256 c) {
  // CHECK-LABEL: test_mm256_fmsubadd_ps
  // CHECK: [[NEG:%.+]] = fsub <8 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %{{.*}}
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK: select <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>, <8 x float> %{{.*}}, <8 x float> %{{.*}}
  return _mm256_fmsubadd_ps(a, b, c);
}

__m256d test_mm256_fmsubadd_pd(__m256d a, __m256d b, __m256d c) {
  // CHECK-LABEL: test_mm256_fmsubadd_pd
  // CHECK: [[NEG:%.+]] = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %{{.+}}
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK: select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %{{.*}}, <4 x double> %{{.*}}
  return _mm256_fmsubadd_pd(a, b, c);
}
