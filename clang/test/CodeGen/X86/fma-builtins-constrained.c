// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +fma -O2 -emit-llvm -o - | FileCheck %s --check-prefixes=COMMON,COMMONIR,UNCONSTRAINED
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +fma -ffp-exception-behavior=strict -O2 -emit-llvm -o - | FileCheck %s --check-prefixes=COMMON,COMMONIR,CONSTRAINED
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +fma -O2 -S -o - | FileCheck %s --check-prefixes=COMMON,CHECK-ASM
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux-gnu -target-feature +fma -O2 -ffp-exception-behavior=strict -S -o - | FileCheck %s --check-prefixes=COMMON,CHECK-ASM

#include <immintrin.h>

__m128 test_mm_fmadd_ps(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fmadd_ps
  // UNCONSTRAINED: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadd213ps
  return _mm_fmadd_ps(a, b, c);
}

__m128d test_mm_fmadd_pd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fmadd_pd
  // UNCONSTRAINED: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CONSTRAINED: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadd213pd
  return _mm_fmadd_pd(a, b, c);
}

__m128 test_mm_fmadd_ss(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fmadd_ss
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // UNCONSTRAINED: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CONSTRAINED: call float @llvm.experimental.constrained.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadd213ss
  // COMMONIR: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fmadd_ss(a, b, c);
}

__m128d test_mm_fmadd_sd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fmadd_sd
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // UNCONSTRAINED: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CONSTRAINED: call double @llvm.experimental.constrained.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadd213sd
  // COMMONIR: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fmadd_sd(a, b, c);
}

__m128 test_mm_fmsub_ps(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fmsub_ps
  // COMMONIR: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // UNCONSTRAINED: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmsub213ps
  return _mm_fmsub_ps(a, b, c);
}

__m128d test_mm_fmsub_pd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fmsub_pd
  // COMMONIR: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // UNCONSTRAINED: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CONSTRAINED: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmsub213pd
  return _mm_fmsub_pd(a, b, c);
}

__m128 test_mm_fmsub_ss(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fmsub_ss
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: [[NEG:%.+]] = fneg float %{{.+}}
  // UNCONSTRAINED: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CONSTRAINED: call float @llvm.experimental.constrained.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmsub213ss
  // COMMONIR: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fmsub_ss(a, b, c);
}

__m128d test_mm_fmsub_sd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fmsub_sd
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: [[NEG:%.+]] = fneg double %{{.+}}
  // UNCONSTRAINED: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CONSTRAINED: call double @llvm.experimental.constrained.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmsub213sd
  // COMMONIR: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fmsub_sd(a, b, c);
}

__m128 test_mm_fnmadd_ps(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fnmadd_ps
  // COMMONIR: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // UNCONSTRAINED: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmadd213ps
  return _mm_fnmadd_ps(a, b, c);
}

__m128d test_mm_fnmadd_pd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fnmadd_pd
  // COMMONIR: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // UNCONSTRAINED: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CONSTRAINED: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmadd213pd
  return _mm_fnmadd_pd(a, b, c);
}

__m128 test_mm_fnmadd_ss(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fnmadd_ss
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: [[NEG:%.+]] = fneg float %{{.+}}
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // UNCONSTRAINED: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CONSTRAINED: call float @llvm.experimental.constrained.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmadd213ss
  // COMMONIR: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fnmadd_ss(a, b, c);
}

__m128d test_mm_fnmadd_sd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fnmadd_sd
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: [[NEG:%.+]] = fneg double %{{.+}}
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // UNCONSTRAINED: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CONSTRAINED: call double @llvm.experimental.constrained.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmadd213sd
  // COMMONIR: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fnmadd_sd(a, b, c);
}

__m128 test_mm_fnmsub_ps(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fnmsub_ps
  // COMMONIR: [[NEG:%.+]] = fneg <4 x float> %{{.+}}
  // COMMONIR: [[NEG2:%.+]] = fneg <4 x float> %{{.+}}
  // UNCONSTRAINED: call <4 x float> @llvm.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CONSTRAINED: call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmsub213ps
  return _mm_fnmsub_ps(a, b, c);
}

__m128d test_mm_fnmsub_pd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fnmsub_pd
  // COMMONIR: [[NEG:%.+]] = fneg <2 x double> %{{.+}}
  // COMMONIR: [[NEG2:%.+]] = fneg <2 x double> %{{.+}}
  // UNCONSTRAINED: call <2 x double> @llvm.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CONSTRAINED: call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmsub213pd
  return _mm_fnmsub_pd(a, b, c);
}

__m128 test_mm_fnmsub_ss(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fnmsub_ss
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: [[NEG:%.+]] = fneg float %{{.+}}
  // COMMONIR: extractelement <4 x float> %{{.*}}, i64 0
  // COMMONIR: [[NEG2:%.+]] = fneg float %{{.+}}
  // UNCONSTRAINED: call float @llvm.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
  // CONSTRAINED: call float @llvm.experimental.constrained.fma.f32(float %{{.*}}, float %{{.*}}, float %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmsub213ss
  // COMMONIR: insertelement <4 x float> %{{.*}}, float %{{.*}}, i64 0
  return _mm_fnmsub_ss(a, b, c);
}

__m128d test_mm_fnmsub_sd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fnmsub_sd
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: [[NEG:%.+]] = fneg double %{{.+}}
  // COMMONIR: extractelement <2 x double> %{{.*}}, i64 0
  // COMMONIR: [[NEG2:%.+]] = fneg double %{{.+}}
  // UNCONSTRAINED: call double @llvm.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}})
  // CONSTRAINED: call double @llvm.experimental.constrained.fma.f64(double %{{.*}}, double %{{.*}}, double %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmsub213sd
  // COMMONIR: insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 0
  return _mm_fnmsub_sd(a, b, c);
}

__m128 test_mm_fmaddsub_ps(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fmaddsub_ps
  // COMMONIR-NOT: fneg
  // COMMONIR: tail call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-ASM: vfmaddsub213ps
  return _mm_fmaddsub_ps(a, b, c);
}

__m128d test_mm_fmaddsub_pd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fmaddsub_pd
  // COMMONIR-NOT: fneg
  // COMMONIR: tail call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK-ASM: vfmaddsub213pd
  return _mm_fmaddsub_pd(a, b, c);
}

__m128 test_mm_fmsubadd_ps(__m128 a, __m128 b, __m128 c) {
  // COMMON-LABEL: test_mm_fmsubadd_ps
  // COMMONIR: [[FNEG:%.+]] = fneg <4 x float> %{{.*}}
  // COMMONIR: tail call <4 x float> @llvm.x86.fma.vfmaddsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[FNEG]])
  // CHECK-ASM: vfmsubadd213ps
  return _mm_fmsubadd_ps(a, b, c);
}

__m128d test_mm_fmsubadd_pd(__m128d a, __m128d b, __m128d c) {
  // COMMON-LABEL: test_mm_fmsubadd_pd
  // COMMONIR: [[FNEG:%.+]] = fneg <2 x double> %{{.*}}
  // COMMONIR: tail call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x double> [[FNEG]])
  // CHECK-ASM: vfmsubadd213pd
  return _mm_fmsubadd_pd(a, b, c);
}

__m256 test_mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
  // COMMON-LABEL: test_mm256_fmadd_ps
  // UNCONSTRAINED: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadd213ps
  return _mm256_fmadd_ps(a, b, c);
}

__m256d test_mm256_fmadd_pd(__m256d a, __m256d b, __m256d c) {
  // COMMON-LABEL: test_mm256_fmadd_pd
  // UNCONSTRAINED: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CONSTRAINED: call <4 x double> @llvm.experimental.constrained.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmadd213pd
  return _mm256_fmadd_pd(a, b, c);
}

__m256 test_mm256_fmsub_ps(__m256 a, __m256 b, __m256 c) {
  // COMMON-LABEL: test_mm256_fmsub_ps
  // COMMONIR: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // UNCONSTRAINED: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmsub213ps
  return _mm256_fmsub_ps(a, b, c);
}

__m256d test_mm256_fmsub_pd(__m256d a, __m256d b, __m256d c) {
  // COMMON-LABEL: test_mm256_fmsub_pd
  // COMMONIR: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // UNCONSTRAINED: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CONSTRAINED: call <4 x double> @llvm.experimental.constrained.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfmsub213pd
  return _mm256_fmsub_pd(a, b, c);
}

__m256 test_mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c) {
  // COMMON-LABEL: test_mm256_fnmadd_ps
  // COMMONIR: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // UNCONSTRAINED: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmadd213ps
  return _mm256_fnmadd_ps(a, b, c);
}

__m256d test_mm256_fnmadd_pd(__m256d a, __m256d b, __m256d c) {
  // COMMON-LABEL: test_mm256_fnmadd_pd
  // COMMONIR: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // UNCONSTRAINED: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CONSTRAINED: call <4 x double> @llvm.experimental.constrained.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmadd213pd
  return _mm256_fnmadd_pd(a, b, c);
}

__m256 test_mm256_fnmsub_ps(__m256 a, __m256 b, __m256 c) {
  // COMMON-LABEL: test_mm256_fnmsub_ps
  // COMMONIR: [[NEG:%.+]] = fneg <8 x float> %{{.*}}
  // COMMONIR: [[NEG2:%.+]] = fneg <8 x float> %{{.*}}
  // UNCONSTRAINED: call <8 x float> @llvm.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CONSTRAINED: call <8 x float> @llvm.experimental.constrained.fma.v8f32(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmsub213ps
  return _mm256_fnmsub_ps(a, b, c);
}

__m256d test_mm256_fnmsub_pd(__m256d a, __m256d b, __m256d c) {
  // COMMON-LABEL: test_mm256_fnmsub_pd
  // COMMONIR: [[NEG:%.+]] = fneg <4 x double> %{{.+}}
  // COMMONIR: [[NEG2:%.+]] = fneg <4 x double> %{{.+}}
  // UNCONSTRAINED: call <4 x double> @llvm.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CONSTRAINED: call <4 x double> @llvm.experimental.constrained.fma.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}}, metadata !{{.*}})
  // CHECK-ASM: vfnmsub213pd
  return _mm256_fnmsub_pd(a, b, c);
}

__m256 test_mm256_fmaddsub_ps(__m256 a, __m256 b, __m256 c) {
  // COMMON-LABEL: test_mm256_fmaddsub_ps
  // COMMONIR-NOT: fneg
  // COMMONIR: tail call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> %{{.*}})
  // CHECK-ASM: vfmaddsub213ps
  return _mm256_fmaddsub_ps(a, b, c);
}

__m256d test_mm256_fmaddsub_pd(__m256d a, __m256d b, __m256d c) {
  // COMMON-LABEL: test_mm256_fmaddsub_pd
  // COMMONIR-NOT: fneg
  // COMMONIR: tail call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> %{{.*}})
  // CHECK-ASM: vfmaddsub213pd
  return _mm256_fmaddsub_pd(a, b, c);
}

__m256 test_mm256_fmsubadd_ps(__m256 a, __m256 b, __m256 c) {
  // COMMON-LABEL: test_mm256_fmsubadd_ps
  // COMMONIR: [[FNEG:%.+]] = fneg <8 x float> %{{.*}}
  // COMMONIR: tail call <8 x float> @llvm.x86.fma.vfmaddsub.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x float> [[FNEG]])
  // CHECK-ASM: vfmsubadd213ps
  return _mm256_fmsubadd_ps(a, b, c);
}

__m256d test_mm256_fmsubadd_pd(__m256d a, __m256d b, __m256d c) {
  // COMMON-LABEL: test_mm256_fmsubadd_pd
  // COMMONIR: [[FNEG:%.+]] = fneg <4 x double> %{{.*}}
  // COMMONIR: tail call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> [[FNEG]])
  // CHECK-ASM: vfmsubadd213pd
  return _mm256_fmsubadd_pd(a, b, c);
}
