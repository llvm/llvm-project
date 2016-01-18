// RUN: %clang_cc1 -ffreestanding -triple x86_64-apple-macosx10.8.0 -target-feature +sse4.1 -emit-llvm %s -o - | FileCheck %s

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

__m128 test_rsqrt_ss(__m128 x) {
  // CHECK: define {{.*}} @test_rsqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rsqrt.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_rsqrt_ss(x);
}

__m128 test_rcp_ss(__m128 x) {
  // CHECK: define {{.*}} @test_rcp_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rcp.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_rcp_ss(x);
}

__m128 test_sqrt_ss(__m128 x) {
  // CHECK: define {{.*}} @test_sqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.sqrt.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_sqrt_ss(x);
}

__m128 test_loadl_pi(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadl_pi
  // CHECK: load <2 x float>, <2 x float>* {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm_loadl_pi(x,y);
}

__m128 test_loadh_pi(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadh_pi
  // CHECK: load <2 x float>, <2 x float>* {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm_loadh_pi(x,y);
}

__m128 test_load_ss(void* y) {
  // CHECK: define {{.*}} @test_load_ss
  // CHECK: load float, float* {{.*}}, align 1{{$}}
  return _mm_load_ss(y);
}

__m128 test_load1_ps(void* y) {
  // CHECK: define {{.*}} @test_load1_ps
  // CHECK: load float, float* {{.*}}, align 1{{$}}
  return _mm_load1_ps(y);
}

void test_store_ss(__m128 x, void* y) {
  // CHECK-LABEL: define void @test_store_ss
  // CHECK: store {{.*}} float* {{.*}}, align 1{{$}}
  _mm_store_ss(y, x);
}

__m128d test_load1_pd(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_load1_pd
  // CHECK: load double, double* {{.*}}, align 1{{$}}
  return _mm_load1_pd(y);
}

__m128d test_loadr_pd(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadr_pd
  // CHECK: load <2 x double>, <2 x double>* {{.*}}, align 16{{$}}
  return _mm_loadr_pd(y);
}

__m128d test_load_sd(void* y) {
  // CHECK: define {{.*}} @test_load_sd
  // CHECK: load double, double* {{.*}}, align 1{{$}}
  return _mm_load_sd(y);
}

__m128d test_loadh_pd(__m128d x, void* y) {
  // CHECK: define {{.*}} @test_loadh_pd
  // CHECK: load double, double* {{.*}}, align 1{{$}}
  return _mm_loadh_pd(x, y);
}

__m128d test_loadl_pd(__m128d x, void* y) {
  // CHECK: define {{.*}} @test_loadl_pd
  // CHECK: load double, double* {{.*}}, align 1{{$}}
  return _mm_loadl_pd(x, y);
}

void test_store_sd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_store_sd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_store_sd(y, x);
}

void test_store1_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_store1_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_store1_pd(y, x);
}

void test_storer_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_storer_pd
  // CHECK: store {{.*}} <2 x double>* {{.*}}, align 16{{$}}
  _mm_storer_pd(y, x);
}

void test_storeh_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_storeh_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_storeh_pd(y, x);
}

void test_storel_pd(__m128d x, void* y) {
  // CHECK-LABEL: define void @test_storel_pd
  // CHECK: store {{.*}} double* {{.*}}, align 1{{$}}
  _mm_storel_pd(y, x);
}

__m128i test_loadl_epi64(void* y) {
  // CHECK: define {{.*}} @test_loadl_epi64
  // CHECK: load i64, i64* {{.*}}, align 1{{$}}
  return _mm_loadl_epi64(y);
}

void test_storel_epi64(__m128i x, void* y) {
  // CHECK-LABEL: define void @test_storel_epi64
  // CHECK: store {{.*}} i64* {{.*}}, align 1{{$}}
  _mm_storel_epi64(y, x);
}

void test_stream_si32(int x, void *y) {
  // CHECK-LABEL: define void @test_stream_si32
  // CHECK: store {{.*}} i32* {{.*}}, align 1, !nontemporal
  _mm_stream_si32(y, x);
}

void test_stream_si64(long long x, void *y) {
  // CHECK-LABEL: define void @test_stream_si64
  // CHECK: store {{.*}} i64* {{.*}}, align 1, !nontemporal
  _mm_stream_si64(y, x);
}

void test_stream_si128(__m128i x, void *y) {
  // CHECK-LABEL: define void @test_stream_si128
  // CHECK: store {{.*}} <2 x i64>* {{.*}}, align 16, !nontemporal
  _mm_stream_si128(y, x);
}

void test_extract_epi16(__m128i __a) {
  // CHECK-LABEL: define void @test_extract_epi16
  // CHECK: [[x:%.*]] = and i32 %{{.*}}, 7
  // CHECK: extractelement <8 x i16> %{{.*}}, i32 [[x]]
  _mm_extract_epi16(__a, 8);
}

__m128 test_mm_cmpeq_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpeq_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 0)
  return _mm_cmpeq_ss(__a, __b);
}

__m128 test_mm_cmplt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmplt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmplt_ss(__a, __b);
}

__m128 test_mm_cmple_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmple_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmple_ss(__a, __b);
}

__m128 test_mm_cmpunord_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpunord_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 3)
  return _mm_cmpunord_ss(__a, __b);
}

__m128 test_mm_cmpneq_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpneq_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_cmpneq_ss(__a, __b);
}

__m128 test_mm_cmpnlt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpnlt_ss(__a, __b);
}

__m128 test_mm_cmpnle_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnle_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnle_ss(__a, __b);
}

__m128 test_mm_cmpord_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpord_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 7)
  return _mm_cmpord_ss(__a, __b);
}

__m128 test_mm_cmpgt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpgt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmpgt_ss(__a, __b);
}

__m128 test_mm_cmpge_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpge_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmpge_ss(__a, __b);
}

__m128 test_mm_cmpngt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpngt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpngt_ss(__a, __b);
}

__m128 test_mm_cmpnge_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnge_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnge_ss(__a, __b);
}

__m128 test_mm_cmpeq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpeq_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 0)
  return _mm_cmpeq_ps(__a, __b);
}

__m128 test_mm_cmplt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmplt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmplt_ps(__a, __b);
}

__m128 test_mm_cmple_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmple_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmple_ps(__a, __b);
}

__m128 test_mm_cmpunord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpunord_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 3)
  return _mm_cmpunord_ps(__a, __b);
}

__m128 test_mm_cmpneq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpneq_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_cmpneq_ps(__a, __b);
}

__m128 test_mm_cmpnlt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpnlt_ps(__a, __b);
}

__m128 test_mm_cmpnle_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnle_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnle_ps(__a, __b);
}

__m128 test_mm_cmpord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpord_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 7)
  return _mm_cmpord_ps(__a, __b);
}

__m128 test_mm_cmpgt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpgt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmpgt_ps(__a, __b);
}

__m128 test_mm_cmpge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpge_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmpge_ps(__a, __b);
}

__m128 test_mm_cmpngt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpngt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpngt_ps(__a, __b);
}

__m128 test_mm_cmpnge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnge_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnge_ps(__a, __b);
}

__m128d test_mm_cmpeq_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpeq_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 0)
  return _mm_cmpeq_sd(__a, __b);
}

__m128d test_mm_cmplt_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmplt_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  return _mm_cmplt_sd(__a, __b);
}

__m128d test_mm_cmple_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmple_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_cmple_sd(__a, __b);
}

__m128d test_mm_cmpunord_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpunord_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 3)
  return _mm_cmpunord_sd(__a, __b);
}

__m128d test_mm_cmpneq_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpneq_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 4)
  return _mm_cmpneq_sd(__a, __b);
}

__m128d test_mm_cmpnlt_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  return _mm_cmpnlt_sd(__a, __b);
}

__m128d test_mm_cmpnle_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpnle_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  return _mm_cmpnle_sd(__a, __b);
}

__m128d test_mm_cmpord_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpord_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 7)
  return _mm_cmpord_sd(__a, __b);
}

__m128d test_mm_cmpgt_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpgt_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  return _mm_cmpgt_sd(__a, __b);
}

__m128d test_mm_cmpge_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpge_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_cmpge_sd(__a, __b);
}

__m128d test_mm_cmpngt_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpngt_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  return _mm_cmpngt_sd(__a, __b);
}

__m128d test_mm_cmpnge_sd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpnge_sd
  // CHECK: @llvm.x86.sse2.cmp.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  return _mm_cmpnge_sd(__a, __b);
}

__m128d test_mm_cmpeq_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpeq_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 0)
  return _mm_cmpeq_pd(__a, __b);
}

__m128d test_mm_cmplt_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmplt_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  return _mm_cmplt_pd(__a, __b);
}

__m128d test_mm_cmple_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmple_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_cmple_pd(__a, __b);
}

__m128d test_mm_cmpunord_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpunord_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 3)
  return _mm_cmpunord_pd(__a, __b);
}

__m128d test_mm_cmpneq_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpneq_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 4)
  return _mm_cmpneq_pd(__a, __b);
}

__m128d test_mm_cmpnlt_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  return _mm_cmpnlt_pd(__a, __b);
}

__m128d test_mm_cmpnle_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpnle_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  return _mm_cmpnle_pd(__a, __b);
}

__m128d test_mm_cmpord_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpord_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 7)
  return _mm_cmpord_pd(__a, __b);
}

__m128d test_mm_cmpgt_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpgt_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 1)
  return _mm_cmpgt_pd(__a, __b);
}

__m128d test_mm_cmpge_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpge_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 2)
  return _mm_cmpge_pd(__a, __b);
}

__m128d test_mm_cmpngt_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpngt_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 5)
  return _mm_cmpngt_pd(__a, __b);
}

__m128d test_mm_cmpnge_pd(__m128d __a, __m128d __b) {
  // CHECK-LABEL: @test_mm_cmpnge_pd
  // CHECK: @llvm.x86.sse2.cmp.pd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i8 6)
  return _mm_cmpnge_pd(__a, __b);
}

__m128 test_mm_slli_si128(__m128 a) {
  // CHECK-LABEL: @test_mm_slli_si128
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  return _mm_slli_si128(a, 5);
}

__m128 test_mm_bslli_si128(__m128 a) {
  // CHECK-LABEL: @test_mm_bslli_si128
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  return _mm_bslli_si128(a, 5);
}

__m128 test_mm_srli_si128(__m128 a) {
  // CHECK-LABEL: @test_mm_srli_si128
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  return _mm_srli_si128(a, 5);
}

__m128 test_mm_bsrli_si128(__m128 a) {
  // CHECK-LABEL: @test_mm_bsrli_si128
  // CHECK: shufflevector <16 x i8> {{.*}}, <16 x i8> {{.*}}, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  return _mm_bsrli_si128(a, 5);
}

__m128 test_mm_undefined_ps() {
  // CHECK-LABEL: @test_mm_undefined_ps
  // CHECK: ret <4 x float> undef
  return _mm_undefined_ps();
}

__m128d test_mm_undefined_pd() {
  // CHECK-LABEL: @test_mm_undefined_pd
  // CHECK: ret <2 x double> undef
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128() {
  // CHECK-LABEL: @test_mm_undefined_si128
  // CHECK: ret <2 x i64> undef
  return _mm_undefined_si128();
}
