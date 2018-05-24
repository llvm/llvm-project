// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -target-feature +avx -target-feature +avx2 -target-feature +avx512f -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -target-feature +avx -target-feature +avx2 -target-feature +avx512f -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK

// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -target-feature +avx -target-feature +avx2 -target-feature +avx512f -emit-llvm -o - -Wall -Werror -fmax-type-align=16 | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -target-feature +avx -target-feature +avx2 -target-feature +avx512f -fno-signed-char -emit-llvm -o - -Wall -Werror -fmax-type-align=16 | FileCheck %s --check-prefix=CHECK

#include <immintrin.h>

// (PR33830) Tests ensure the correct alignment of non-temporal load/stores on darwin targets where fmax-type-align is set to 16.

//
// 128-bit vectors
//

void test_mm_stream_pd(double* A, __m128d B) {
  // CHECK-LABEL: test_mm_stream_pd
  // CHECK: store <2 x double> %{{.*}}, <2 x double>* %{{.*}}, align 16, !nontemporal
  _mm_stream_pd(A, B);
}

void test_mm_stream_ps(float* A, __m128 B) {
  // CHECK16-LABEL: test_mm_stream_ps
  // CHECK16: store <4 x float> %{{.*}}, <4 x float>* %{{.*}}, align 16, !nontemporal
  _mm_stream_ps(A, B);
}

void test_mm_stream_si128(__m128i* A, __m128i B) {
  // CHECK-LABEL: test_mm_stream_si128
  // CHECK: store <2 x i64> %{{.*}}, <2 x i64>* %{{.*}}, align 16, !nontemporal
  _mm_stream_si128(A, B);
}

__m128i test_mm_stream_load_si128(__m128i const *A) {
  // CHECK-LABEL: test_mm_stream_load_si128
  // CHECK: load <2 x i64>, <2 x i64>* %{{.*}}, align 16, !nontemporal
  return _mm_stream_load_si128(A);
}

//
// 256-bit vectors
//

void test_mm256_stream_pd(double* A, __m256d B) {
  // CHECK-LABEL: test_mm256_stream_pd
  // CHECK: store <4 x double> %{{.*}}, <4 x double>* %{{.*}}, align 32, !nontemporal
  _mm256_stream_pd(A, B);
}

void test_mm256_stream_ps(float* A, __m256 B) {
  // CHECK-LABEL: test_mm256_stream_ps
  // CHECK: store <8 x float> %{{.*}}, <8 x float>* %{{.*}}, align 32, !nontemporal
  _mm256_stream_ps(A, B);
}

void test_mm256_stream_si256(__m256i* A, __m256i B) {
  // CHECK-LABEL: test_mm256_stream_si256
  // CHECK: store <4 x i64> %{{.*}}, <4 x i64>* %{{.*}}, align 32, !nontemporal
  _mm256_stream_si256(A, B);
}

__m256i test_mm256_stream_load_si256(__m256i const *A) {
  // CHECK-LABEL: test_mm256_stream_load_si256
  // CHECK: load <4 x i64>, <4 x i64>* %{{.*}}, align 32, !nontemporal
  return _mm256_stream_load_si256(A);
}

//
// 512-bit vectors
//

void test_mm512_stream_pd(double* A, __m512d B) {
  // CHECK-LABEL: test_mm512_stream_pd
  // CHECK: store <8 x double> %{{.*}}, <8 x double>* %{{.*}}, align 64, !nontemporal
  _mm512_stream_pd(A, B);
}

void test_mm512_stream_ps(float* A, __m512 B) {
  // CHECK-LABEL: test_mm512_stream_ps
  // CHECK: store <16 x float> %{{.*}}, <16 x float>* %{{.*}}, align 64, !nontemporal
  _mm512_stream_ps(A, B);
}

void test_mm512_stream_si512(__m512i* A, __m512i B) {
  // CHECK-LABEL: test_mm512_stream_si512
  // CHECK: store <8 x i64> %{{.*}}, <8 x i64>* %{{.*}}, align 64, !nontemporal
  _mm512_stream_si512(A, B);
}

__m512i test_mm512_stream_load_si512(void *A) {
  // CHECK-LABEL: test_mm512_stream_load_si512
  // CHECK: load <8 x i64>, <8 x i64>* %{{.*}}, align 64, !nontemporal
  return _mm512_stream_load_si512(A);
}
