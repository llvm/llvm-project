// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4a -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4a -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4a -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4a -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <x86intrin.h>

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse4a-intrinsics-fast-isel.ll

__m128i test_mm_extracti_si64(__m128i x) {
  // CHECK-LABEL: test_mm_extracti_si64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %{{[^,]+}}, i8 3, i8 2)
  return _mm_extracti_si64(x, 3, 2);
}

__m128i test_mm_extract_si64(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_extract_si64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %{{[^,]+}}, <16 x i8> %{{[^,]+}})
  return _mm_extract_si64(x, y);
}

__m128i test_mm_inserti_si64(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_inserti_si64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %{{[^,]+}}, <2 x i64> %{{[^,]+}}, i8 5, i8 6)
  return _mm_inserti_si64(x, y, 5, 6);
}

__m128i test_mm_insert_si64(__m128i x, __m128i y) {
  // CHECK-LABEL: test_mm_insert_si64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.sse4a.insertq(<2 x i64> %{{[^,]+}}, <2 x i64> %{{[^,]+}})
  return _mm_insert_si64(x, y);
}

void test_mm_stream_sd(double *p, __m128d a) {
  // CHECK-LABEL: test_mm_stream_sd
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: store double %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
   _mm_stream_sd(p, a);
}

void test_mm_stream_sd_void(void *p, __m128d a) {
  // CHECK-LABEL: test_mm_stream_sd_void
  // CHECK: extractelement <2 x double> %{{.*}}, i64 0
  // CHECK: store double %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
   _mm_stream_sd(p, a);
}

void test_mm_stream_ss(float *p, __m128 a) {
  // CHECK-LABEL: test_mm_stream_ss
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: store float %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
  _mm_stream_ss(p, a);
}

void test_mm_stream_s_void(void *p, __m128 a) {
  // CHECK-LABEL: test_mm_stream_s_void
  // CHECK: extractelement <4 x float> %{{.*}}, i64 0
  // CHECK: store float %{{.*}}, ptr %{{.*}}, align 1, !nontemporal
  _mm_stream_ss(p, a);
}
