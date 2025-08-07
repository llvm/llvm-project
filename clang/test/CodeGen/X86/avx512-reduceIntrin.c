// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>
#include "builtin_test_helpers.h"

long long test_mm512_reduce_add_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_add_epi64(
// CHECK:    call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_add_epi64(__W);
}
TEST_CONSTEXPR(_mm512_reduce_add_epi64((__m512i)(__v8di){-4, -3, -2, -1, 0, 1, 2, 3}) == -4);

long long test_mm512_reduce_mul_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_mul_epi64(
// CHECK:    call i64 @llvm.vector.reduce.mul.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_mul_epi64(__W);
}
TEST_CONSTEXPR(_mm512_reduce_mul_epi64((__m512i)(__v8di){1, 2, 3, 4, 5, 6, 7, 8}) == 40320);

long long test_mm512_reduce_or_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_or_epi64(
// CHECK:    call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_or_epi64(__W);
}
TEST_CONSTEXPR(_mm512_reduce_or_epi64((__m512i)(__v8di){0x100, 0x200, 0x400, 0x800, 0, 0, 0, 0}) == 0xF00);

long long test_mm512_reduce_and_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_and_epi64(
// CHECK:    call i64 @llvm.vector.reduce.and.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_and_epi64(__W);
}
TEST_CONSTEXPR(_mm512_reduce_and_epi64((__m512i)(__v8di){0xFFFF, 0xFF00, 0x00FF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFF00, 0x00FF}) == 0x0000);

long long test_mm512_mask_reduce_add_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_epi64(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_add_epi64(__M, __W);
}

long long test_mm512_mask_reduce_mul_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_epi64(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.mul.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_mul_epi64(__M, __W);
}

long long test_mm512_mask_reduce_and_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_and_epi64(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.and.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_and_epi64(__M, __W);
}

long long test_mm512_mask_reduce_or_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_or_epi64(
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_or_epi64(__M, __W);
}

int test_mm512_reduce_add_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_add_epi32(
// CHECK:    call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_add_epi32(__W);
}
TEST_CONSTEXPR(_mm512_reduce_add_epi32((__m512i)(__v16si){-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}) == -8);

int test_mm512_reduce_mul_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_mul_epi32(
// CHECK:    call i32 @llvm.vector.reduce.mul.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_mul_epi32(__W);
}
TEST_CONSTEXPR(_mm512_reduce_mul_epi32((__m512i)(__v16si){1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 1, -3, 1, 1}) == -36);

int test_mm512_reduce_or_epi32(__m512i __W){
// CHECK:    call i32 @llvm.vector.reduce.or.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_or_epi32(__W);
}
TEST_CONSTEXPR(_mm512_reduce_or_epi32((__m512i)(__v16si){0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0, 0, 0, 0, 0, 0, 0, 0}) == 0xFF);

int test_mm512_reduce_and_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_and_epi32(
// CHECK:    call i32 @llvm.vector.reduce.and.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_and_epi32(__W);
}
TEST_CONSTEXPR(_mm512_reduce_and_epi32((__m512i)(__v16si){0xFF, 0xF0, 0x0F, 0xFF, 0xFF, 0xFF, 0xF0, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0, 0x0F, 0x0F}) == 0x00);

int test_mm512_mask_reduce_add_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_epi32(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_add_epi32(__M, __W);
}

int test_mm512_mask_reduce_mul_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_epi32(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.mul.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_mul_epi32(__M, __W);
}

int test_mm512_mask_reduce_and_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_and_epi32(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.and.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_and_epi32(__M, __W);
}

int test_mm512_mask_reduce_or_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_or_epi32(
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.or.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_or_epi32(__M, __W);
}

double test_mm512_reduce_add_pd(__m512d __W, double ExtraAddOp){
// CHECK-LABEL: @test_mm512_reduce_add_pd(
// CHECK-NOT: reassoc
// CHECK:    call reassoc double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})
// CHECK-NOT: reassoc
  return _mm512_reduce_add_pd(__W) + ExtraAddOp;
}

double test_mm512_reduce_mul_pd(__m512d __W, double ExtraMulOp){
// CHECK-LABEL: @test_mm512_reduce_mul_pd(
// CHECK-NOT: reassoc
// CHECK:    call reassoc double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})
// CHECK-NOT: reassoc
  return _mm512_reduce_mul_pd(__W) * ExtraMulOp;
}

float test_mm512_reduce_add_ps(__m512 __W){
// CHECK-LABEL: @test_mm512_reduce_add_ps(
// CHECK:    call reassoc float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})
  return _mm512_reduce_add_ps(__W);
}

float test_mm512_reduce_mul_ps(__m512 __W){
// CHECK-LABEL: @test_mm512_reduce_mul_ps(
// CHECK:    call reassoc float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})
  return _mm512_reduce_mul_ps(__W);
}

double test_mm512_mask_reduce_add_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_pd(
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    call reassoc double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})
  return _mm512_mask_reduce_add_pd(__M, __W);
}

double test_mm512_mask_reduce_mul_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_pd(
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    call reassoc double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})
  return _mm512_mask_reduce_mul_pd(__M, __W);
}

float test_mm512_mask_reduce_add_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_ps(
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}
// CHECK:    call reassoc float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})
  return _mm512_mask_reduce_add_ps(__M, __W);
}

float test_mm512_mask_reduce_mul_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_ps(
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> {{.*}}, <16 x float> %{{.*}}
// CHECK:    call reassoc float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})
  return _mm512_mask_reduce_mul_ps(__M, __W);
}
