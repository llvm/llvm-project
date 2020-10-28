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

double test_mm512_reduce_max_pd(__m512d __W){
// CHECK-LABEL: @test_mm512_reduce_max_pd(
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
  return _mm512_reduce_max_pd(__W); 
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

double test_mm512_reduce_min_pd(__m512d __W){
// CHECK-LABEL: @test_mm512_reduce_min_pd(
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
  return _mm512_reduce_min_pd(__W); 
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
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x double> @llvm.x86.avx.max.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double>  %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    call <2 x double> @llvm.x86.sse2.max.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
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
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x double> @llvm.x86.avx.min.pd.256(<4 x double> %{{.*}}, <4 x double> %{{.*}})
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 0, i32 1>
// CHECK:    shufflevector <4 x double> %{{.*}}, <4 x double> undef, <2 x i32> <i32 2, i32 3>
// CHECK:    call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    shufflevector <2 x double> %{{.*}}, <2 x double>  %{{.*}}, <2 x i32> <i32 1, i32 0>
// CHECK:    call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}})
// CHECK:    extractelement <2 x double> %{{.*}}, i32 0
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
// CHECK-LABEL: define float @test_mm512_reduce_max_ps(
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
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
// CHECK-LABEL: define float @test_mm512_reduce_min_ps(
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
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
// CHECK-LABEL: define float @test_mm512_mask_reduce_max_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
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
// CHECK-LABEL: define float @test_mm512_mask_reduce_min_ps(
// CHECK:    bitcast i16 %{{.*}} to <16 x i1>
// CHECK:    select <16 x i1> %{{.*}}, <16 x float> %{{.*}}, <16 x float> %{{.*}}
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x double> %{{.*}}, <8 x double> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <8 x float> @llvm.x86.avx.min.ps.256(<8 x float> %{{.*}}, <8 x float> %{{.*}})
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK:    shufflevector <8 x float> %{{.*}}, <8 x float> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK:    call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
// CHECK:    call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
// CHECK:    call <4 x float> @llvm.x86.sse.min.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK:    extractelement <4 x float> %{{.*}}, i32 0
  return _mm512_mask_reduce_min_ps(__M, __W); 
}

