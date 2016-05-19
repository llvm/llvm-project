// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

//
// Test LLVM IR codegen of shuffle instructions
//

__m256 test__mm256_loadu_ps(void* p) {
  // CHECK: load <8 x float>, <8 x float>* %{{.*}}, align 1
  return _mm256_loadu_ps(p);
}

__m256d test__mm256_loadu_pd(void* p) {
  // CHECK: load <4 x double>, <4 x double>* %{{.*}}, align 1
  return _mm256_loadu_pd(p);
}

__m256i test__mm256_loadu_si256(void* p) {
  // CHECK: load <4 x i64>, <4 x i64>* %{{.+}}, align 1
  return _mm256_loadu_si256(p);
}

int test_extract_epi32(__m256i __a) {
  // CHECK-LABEL: @test_extract_epi32
  // CHECK: [[SHIFT1:%[^ ]+]] = and i32 %{{.*}}, 7
  // CHECK: extractelement <8 x i32> %{{.*}}, i32 [[SHIFT1]]
  return _mm256_extract_epi32(__a, 8);
}

int test_extract_epi16(__m256i __a) {
  // CHECK-LABEL: @test_extract_epi16
  // CHECK: [[SHIFT2:%[^ ]+]] = and i32 %{{.*}}, 15
  // CHECK: extractelement <16 x i16> %{{.*}}, i32 [[SHIFT2]]
  return _mm256_extract_epi16(__a, 16);
}

int test_extract_epi8(__m256i __a) {
  // CHECK-LABEL: @test_extract_epi8
  // CHECK: [[SHIFT3:%[^ ]+]] = and i32 %{{.*}}, 31
  // CHECK: extractelement <32 x i8> %{{.*}}, i32 [[SHIFT3]]
  return _mm256_extract_epi8(__a, 32);
}

__m256d test_256_blend_pd(__m256d __a, __m256d __b) {
  // CHECK-LABEL: @test_256_blend_pd
  // CHECK: shufflevector <4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  return _mm256_blend_pd(__a, __b, 0x35);
}

__m256 test_256_blend_ps(__m256 __a, __m256 __b) {
  // CHECK-LABEL: @test_256_blend_ps
  // CHECK: shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  return _mm256_blend_ps(__a, __b, 0x35);
}

__m256i test_256_insert_epi8(__m256i __a) {
  // CHECK-LABEL: @test_256_insert_epi8
  // CHECK: insertelement <32 x i8> {{.*}}, i8 {{.*}}, i32 {{.*}}
  return _mm256_insert_epi8(__a, 42, 3);
}

__m256i test_256_insert_epi16(__m256i __a) {
  // CHECK-LABEL: @test_256_insert_epi16
  // CHECK: insertelement <16 x i16> {{.*}}, i16 {{.*}}, i32 {{.*}}
  return _mm256_insert_epi16(__a, 42, 3);
}

__m256i test_256_insert_epi32(__m256i __a) {
  // CHECK-LABEL: @test_256_insert_epi32
  // CHECK: insertelement <8 x i32> {{.*}}, i32 {{.*}}, i32 {{.*}}
  return _mm256_insert_epi32(__a, 42, 3);
}

__m256i test_256_insert_epi64(__m256i __a) {
  // CHECK-LABEL: @test_256_insert_epi64
  // CHECK: insertelement <4 x i64> {{.*}}, i64 {{.*}}, i32 {{.*}}
  return _mm256_insert_epi64(__a, 42, 3);
}

__m256 test_mm256_undefined_ps() {
  // CHECK-LABEL: @test_mm256_undefined_ps
  // CHECK: ret <8 x float> undef
  return _mm256_undefined_ps();
}

__m256d test_mm256_undefined_pd() {
  // CHECK-LABEL: @test_mm256_undefined_pd
  // CHECK: ret <4 x double> undef
  return _mm256_undefined_pd();
}

__m256i test_mm256_undefined_si256() {
  // CHECK-LABEL: @test_mm256_undefined_si256
  // CHECK: ret <4 x i64> undef
  return _mm256_undefined_si256();
}
