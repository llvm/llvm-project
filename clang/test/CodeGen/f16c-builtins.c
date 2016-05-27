// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +f16c -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

float test_cvtsh_ss(unsigned short a) {
  // CHECK-LABEL: test_cvtsh_ss
  // CHECK: insertelement <8 x i16> undef, i16 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 1
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 2
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 3
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 4
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 5
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 6
  // CHECK: insertelement <8 x i16> %{{.*}}, i16 0, i32 7
  // CHECK: call <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16> %{{.*}})
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  return _cvtsh_ss(a);
}

unsigned short test_cvtss_sh(float a) {
  // CHECK-LABEL: test_cvtss_sh
  // CHECK: insertelement <4 x float> undef, float %{{.*}}, i32 0
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 1
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 2
  // CHECK: insertelement <4 x float> %{{.*}}, float 0.000000e+00, i32 3
  // CHECK: call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %{{.*}}, i32 0)
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 0
  return _cvtss_sh(a, 0);
}

__m128 test_mm_cvtph_ps(__m128i a) {
  // CHECK-LABEL: test_mm_cvtph_ps
  // CHECK: call <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16> %{{.*}})
  return _mm_cvtph_ps(a);
}

__m256 test_mm256_cvtph_ps(__m128i a) {
  // CHECK-LABEL: test_mm256_cvtph_ps
  // CHECK: call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %{{.*}})
  return _mm256_cvtph_ps(a);
}

__m128i test_mm_cvtps_ph(__m128 a) {
  // CHECK-LABEL: test_mm_cvtps_ph
  // CHECK: call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %{{.*}}, i32 0)
  return _mm_cvtps_ph(a, 0);
}

__m128i test_mm256_cvtps_ph(__m256 a) {
  // CHECK-LABEL: test_mm256_cvtps_ph
  // CHECK: call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %{{.*}}, i32 0)
  return _mm256_cvtps_ph(a, 0);
}
