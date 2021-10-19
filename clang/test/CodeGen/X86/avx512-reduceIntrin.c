// RUN: %clang_cc1 -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

long long test_mm512_reduce_add_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_add_epi64(
// CHECK:    call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_add_epi64(__W);
}

long long test_mm512_reduce_mul_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_mul_epi64(
// CHECK:    call i64 @llvm.vector.reduce.mul.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_mul_epi64(__W);
}

long long test_mm512_reduce_or_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_or_epi64(
// CHECK:    call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_or_epi64(__W);
}

long long test_mm512_reduce_and_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_and_epi64(
// CHECK:    call i64 @llvm.vector.reduce.and.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_and_epi64(__W);
}

long long test_mm512_mask_reduce_add_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.add.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_add_epi64(__M, __W);
}

long long test_mm512_mask_reduce_mul_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.mul.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_mul_epi64(__M, __W);
}

long long test_mm512_mask_reduce_and_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_and_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.and.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_and_epi64(__M, __W);
}

long long test_mm512_mask_reduce_or_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_or_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_or_epi64(__M, __W);
}

int test_mm512_reduce_add_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_add_epi32(
// CHECK:    call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_add_epi32(__W);
}

int test_mm512_reduce_mul_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_mul_epi32(
// CHECK:    call i32 @llvm.vector.reduce.mul.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_mul_epi32(__W);
}

int test_mm512_reduce_or_epi32(__m512i __W){
// CHECK:    call i32 @llvm.vector.reduce.or.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_or_epi32(__W);
}

int test_mm512_reduce_and_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_and_epi32(
// CHECK:    call i32 @llvm.vector.reduce.and.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_and_epi32(__W);
}

int test_mm512_mask_reduce_add_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_add_epi32(__M, __W);
}

int test_mm512_mask_reduce_mul_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.mul.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_mul_epi32(__M, __W);
}

int test_mm512_mask_reduce_and_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_and_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.and.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_and_epi32(__M, __W);
}

int test_mm512_mask_reduce_or_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_or_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
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
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    call reassoc double @llvm.vector.reduce.fadd.v8f64(double -0.000000e+00, <8 x double> %{{.*}})
  return _mm512_mask_reduce_add_pd(__M, __W);
}

double test_mm512_mask_reduce_mul_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_pd(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    call reassoc double @llvm.vector.reduce.fmul.v8f64(double 1.000000e+00, <8 x double> %{{.*}})
  return _mm512_mask_reduce_mul_pd(__M, __W);
}

float test_mm512_mask_reduce_add_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_add_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> {{.*}}, <16 x float> {{.*}}
// CHECK:    call reassoc float @llvm.vector.reduce.fadd.v16f32(float -0.000000e+00, <16 x float> %{{.*}})
  return _mm512_mask_reduce_add_ps(__M, __W);
}

float test_mm512_mask_reduce_mul_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_mul_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> {{.*}}, <16 x float> %{{.*}}
// CHECK:    call reassoc float @llvm.vector.reduce.fmul.v16f32(float 1.000000e+00, <16 x float> %{{.*}})
  return _mm512_mask_reduce_mul_ps(__M, __W);
}
