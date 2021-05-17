// RUN: %clang_cc1 -fexperimental-new-pass-manager -ffreestanding %s -O0 -triple=x86_64-apple-darwin -target-cpu skylake-avx512 -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

long long test_mm512_reduce_max_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_max_epi64(
// CHECK:    call i64 @llvm.vector.reduce.smax.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_max_epi64(__W);
}

unsigned long long test_mm512_reduce_max_epu64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_max_epu64(
// CHECK:    call i64 @llvm.vector.reduce.umax.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_max_epu64(__W);
}

double test_mm512_reduce_max_pd(__m512d __W, double ExtraAddOp){
// CHECK-LABEL: @test_mm512_reduce_max_pd(
// CHECK-NOT: nnan
// CHECK:    call nnan double @llvm.vector.reduce.fmax.v8f64(<8 x double> %{{.*}})
// CHECK-NOT: nnan
  return _mm512_reduce_max_pd(__W) + ExtraAddOp;
}

long long test_mm512_reduce_min_epi64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_min_epi64(
// CHECK:    call i64 @llvm.vector.reduce.smin.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_min_epi64(__W);
}

unsigned long long test_mm512_reduce_min_epu64(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_min_epu64(
// CHECK:    call i64 @llvm.vector.reduce.umin.v8i64(<8 x i64> %{{.*}})
  return _mm512_reduce_min_epu64(__W);
}

double test_mm512_reduce_min_pd(__m512d __W, double ExtraMulOp){
// CHECK-LABEL: @test_mm512_reduce_min_pd(
// CHECK-NOT: nnan
// CHECK:    call nnan double @llvm.vector.reduce.fmin.v8f64(<8 x double> %{{.*}})
// CHECK-NOT: nnan
  return _mm512_reduce_min_pd(__W) * ExtraMulOp;
}

long long test_mm512_mask_reduce_max_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_max_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.smax.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_max_epi64(__M, __W);
}

unsigned long test_mm512_mask_reduce_max_epu64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_max_epu64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.umax.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_max_epu64(__M, __W);
}

double test_mm512_mask_reduce_max_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_max_pd(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    call nnan double @llvm.vector.reduce.fmax.v8f64(<8 x double> %{{.*}})
  return _mm512_mask_reduce_max_pd(__M, __W);
}

long long test_mm512_mask_reduce_min_epi64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_min_epi64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.smin.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_min_epi64(__M, __W);
}

unsigned long long test_mm512_mask_reduce_min_epu64(__mmask8 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_min_epu64(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
// CHECK:    call i64 @llvm.vector.reduce.umin.v8i64(<8 x i64> %{{.*}})
  return _mm512_mask_reduce_min_epu64(__M, __W);
}

double test_mm512_mask_reduce_min_pd(__mmask8 __M, __m512d __W){
// CHECK-LABEL: @test_mm512_mask_reduce_min_pd(
// CHECK:    bitcast i8 %{{.*}} to <8 x i1>
// CHECK:    select <8 x i1> %{{.*}}, <8 x double> %{{.*}}, <8 x double> %{{.*}}
// CHECK:    call nnan double @llvm.vector.reduce.fmin.v8f64(<8 x double> %{{.*}})
  return _mm512_mask_reduce_min_pd(__M, __W);
}

int test_mm512_reduce_max_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_max_epi32(
// CHECK:    call i32 @llvm.vector.reduce.smax.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_max_epi32(__W);
}

unsigned int test_mm512_reduce_max_epu32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_max_epu32(
// CHECK:    call i32 @llvm.vector.reduce.umax.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_max_epu32(__W);
}

float test_mm512_reduce_max_ps(__m512 __W){
// CHECK-LABEL: @test_mm512_reduce_max_ps(
// CHECK:    call nnan float @llvm.vector.reduce.fmax.v16f32(<16 x float> %{{.*}})
  return _mm512_reduce_max_ps(__W);
}

int test_mm512_reduce_min_epi32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_min_epi32(
// CHECK:    call i32 @llvm.vector.reduce.smin.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_min_epi32(__W);
}

unsigned int test_mm512_reduce_min_epu32(__m512i __W){
// CHECK-LABEL: @test_mm512_reduce_min_epu32(
// CHECK:    call i32 @llvm.vector.reduce.umin.v16i32(<16 x i32> %{{.*}})
  return _mm512_reduce_min_epu32(__W);
}

float test_mm512_reduce_min_ps(__m512 __W){
// CHECK-LABEL: @test_mm512_reduce_min_ps(
// CHECK:    call nnan float @llvm.vector.reduce.fmin.v16f32(<16 x float> %{{.*}})
  return _mm512_reduce_min_ps(__W);
}

int test_mm512_mask_reduce_max_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_max_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.smax.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_max_epi32(__M, __W);
}

unsigned int test_mm512_mask_reduce_max_epu32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_max_epu32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.umax.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_max_epu32(__M, __W);
}

float test_mm512_mask_reduce_max_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_max_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
// CHECK:    call nnan float @llvm.vector.reduce.fmax.v16f32(<16 x float> %{{.*}})
  return _mm512_mask_reduce_max_ps(__M, __W);
}

int test_mm512_mask_reduce_min_epi32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_min_epi32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.smin.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_min_epi32(__M, __W);
}

unsigned int test_mm512_mask_reduce_min_epu32(__mmask16 __M, __m512i __W){
// CHECK-LABEL: @test_mm512_mask_reduce_min_epu32(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
// CHECK:    call i32 @llvm.vector.reduce.umin.v16i32(<16 x i32> %{{.*}})
  return _mm512_mask_reduce_min_epu32(__M, __W);
}

float test_mm512_mask_reduce_min_ps(__mmask16 __M, __m512 __W){
// CHECK-LABEL: @test_mm512_mask_reduce_min_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
// CHECK:    call nnan float @llvm.vector.reduce.fmin.v16f32(<16 x float> %{{.*}})
  return _mm512_mask_reduce_min_ps(__M, __W);
}

