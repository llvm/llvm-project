// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse3 -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse3-intrinsics-fast-isel.ll

__m128d test_mm_addsub_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_addsub_pd
  // CHECK: call <2 x double> @llvm.x86.sse3.addsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_addsub_pd(A, B);
}

__m128 test_mm_addsub_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_addsub_ps
  // CHECK: call <4 x float> @llvm.x86.sse3.addsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_addsub_ps(A, B);
}

__m128d test_mm_hadd_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_hadd_pd
  // CHECK: call <2 x double> @llvm.x86.sse3.hadd.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_hadd_pd(A, B);
}

__m128 test_mm_hadd_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_hadd_ps
  // CHECK: call <4 x float> @llvm.x86.sse3.hadd.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_hadd_ps(A, B);
}

__m128d test_mm_hsub_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_hsub_pd
  // CHECK: call <2 x double> @llvm.x86.sse3.hsub.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
  return _mm_hsub_pd(A, B);
}

__m128 test_mm_hsub_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_hsub_ps
  // CHECK: call <4 x float> @llvm.x86.sse3.hsub.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  return _mm_hsub_ps(A, B);
}

__m128i test_mm_lddqu_si128(__m128i const* P) {
  // CHECK-LABEL: test_mm_lddqu_si128
  // CHECK: call <16 x i8> @llvm.x86.sse3.ldu.dq(i8* %{{.*}})
  return _mm_lddqu_si128(P);
}

__m128d test_mm_loaddup_pd(double const* P) {
  // CHECK-LABEL: test_mm_loaddup_pd
  // CHECK: load double*
  // CHECK: insertelement <2 x double> undef, double %{{.*}}, i32 0
  // CHECK: insertelement <2 x double> %{{.*}}, double %{{.*}}, i32 1
  return _mm_loaddup_pd(P);
}

__m128d test_mm_movedup_pd(__m128d A) {
  // CHECK-LABEL: test_mm_movedup_pd
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  return _mm_movedup_pd(A);
}

__m128 test_mm_movehdup_ps(__m128 A) {
  // CHECK-LABEL: test_mm_movehdup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 1, i32 3, i32 3>
  return _mm_movehdup_ps(A);
}

__m128 test_mm_moveldup_ps(__m128 A) {
  // CHECK-LABEL: test_mm_moveldup_ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  return _mm_moveldup_ps(A);
}
