// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m512d test_mm512_sqrt_pd(__m512d a)
{
  // CHECK-LABEL: @test_mm512_sqrt_pd
  // CHECK: @llvm.x86.avx512.mask.sqrt.pd.512
  return _mm512_sqrt_pd(a);
}

__m512 test_mm512_sqrt_ps(__m512 a)
{
  // CHECK-LABEL: @test_mm512_sqrt_ps
  // CHECK: @llvm.x86.avx512.mask.sqrt.ps.512
  return _mm512_sqrt_ps(a);
}

__m512d test_mm512_rsqrt14_pd(__m512d a)
{
  // CHECK-LABEL: @test_mm512_rsqrt14_pd
  // CHECK: @llvm.x86.avx512.rsqrt14.pd.512
  return _mm512_rsqrt14_pd(a);
}

__m512 test_mm512_rsqrt14_ps(__m512 a)
{
  // CHECK-LABEL: @test_mm512_rsqrt14_ps
  // CHECK: @llvm.x86.avx512.rsqrt14.ps.512
  return _mm512_rsqrt14_ps(a);
}

__m512 test_mm512_add_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: @test_mm512_add_ps
  // CHECK: fadd <16 x float>
  return _mm512_add_ps(a, b);
}

__m512d test_mm512_add_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_add_pd
  // CHECK: fadd <8 x double>
  return _mm512_add_pd(a, b);
}

__m512 test_mm512_mul_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: @test_mm512_mul_ps
  // CHECK: fmul <16 x float>
  return _mm512_mul_ps(a, b);
}

__m512d test_mm512_mul_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_mul_pd
  // CHECK: fmul <8 x double>
  return _mm512_mul_pd(a, b);
}

void test_mm512_storeu_ps(void *p, __m512 a)
{
  // CHECK-LABEL: @test_mm512_storeu_ps
  // CHECK: @llvm.x86.avx512.mask.storeu.ps.512
  _mm512_storeu_ps(p, a);
}

void test_mm512_storeu_pd(void *p, __m512d a)
{
  // CHECK-LABEL: @test_mm512_storeu_pd
  // CHECK: @llvm.x86.avx512.mask.storeu.pd.512
  _mm512_storeu_pd(p, a);
}

void test_mm512_mask_store_ps(void *p, __m512 a, __mmask16 m)
{
  // CHECK-LABEL: @test_mm512_mask_store_ps
  // CHECK: @llvm.x86.avx512.mask.store.ps.512
  _mm512_mask_store_ps(p, m, a);
}

void test_mm512_store_ps(void *p, __m512 a)
{
  // CHECK-LABEL: @test_mm512_store_ps
  // CHECK: store <16 x float>
  _mm512_store_ps(p, a);
}

void test_mm512_mask_store_pd(void *p, __m512d a, __mmask8 m)
{
  // CHECK-LABEL: @test_mm512_mask_store_pd
  // CHECK: @llvm.x86.avx512.mask.store.pd.512
  _mm512_mask_store_pd(p, m, a);
}

void test_mm512_store_pd(void *p, __m512d a)
{
  // CHECK-LABEL: @test_mm512_store_pd
  // CHECK: store <8 x double>
  _mm512_store_pd(p, a);
}

__m512 test_mm512_loadu_ps(void *p)
{
  // CHECK-LABEL: @test_mm512_loadu_ps
  // CHECK: load <16 x float>, <16 x float>* {{.*}}, align 1{{$}}
  return _mm512_loadu_ps(p);
}

__m512d test_mm512_loadu_pd(void *p)
{
  // CHECK-LABEL: @test_mm512_loadu_pd
  // CHECK: load <8 x double>, <8 x double>* {{.*}}, align 1{{$}}
  return _mm512_loadu_pd(p);
}

__m512 test_mm512_maskz_load_ps(void *p, __mmask16 m)
{
  // CHECK-LABEL: @test_mm512_maskz_load_ps
  // CHECK: @llvm.x86.avx512.mask.load.ps.512
  return _mm512_maskz_load_ps(m, p);
}

__m512 test_mm512_load_ps(void *p)
{
  // CHECK-LABEL: @test_mm512_load_ps
  // CHECK: @llvm.x86.avx512.mask.load.ps.512
  return _mm512_load_ps(p);
}

__m512d test_mm512_maskz_load_pd(void *p, __mmask8 m)
{
  // CHECK-LABEL: @test_mm512_maskz_load_pd
  // CHECK: @llvm.x86.avx512.mask.load.pd.512
  return _mm512_maskz_load_pd(m, p);
}

__m512d test_mm512_load_pd(void *p)
{
  // CHECK-LABEL: @test_mm512_load_pd
  // CHECK: @llvm.x86.avx512.mask.load.pd.512
  return _mm512_load_pd(p);
}

__m512d test_mm512_set1_pd(double d)
{
  // CHECK-LABEL: @test_mm512_set1_pd
  // CHECK: insertelement <8 x double> {{.*}}, i32 0
  // CHECK: insertelement <8 x double> {{.*}}, i32 1
  // CHECK: insertelement <8 x double> {{.*}}, i32 2
  // CHECK: insertelement <8 x double> {{.*}}, i32 3
  // CHECK: insertelement <8 x double> {{.*}}, i32 4
  // CHECK: insertelement <8 x double> {{.*}}, i32 5
  // CHECK: insertelement <8 x double> {{.*}}, i32 6
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_set1_pd(d);
}

__m512d test_mm512_castpd256_pd512(__m256d a)
{
  // CHECK-LABEL: @test_mm512_castpd256_pd512
  // CHECK: shufflevector <4 x double> {{.*}} <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  return _mm512_castpd256_pd512(a);
}

__mmask16 test_mm512_knot(__mmask16 a)
{
  // CHECK-LABEL: @test_mm512_knot
  // CHECK: @llvm.x86.avx512.knot.w
  return _mm512_knot(a);
}

__m512i test_mm512_alignr_epi32(__m512i a, __m512i b)
{
  // CHECK-LABEL: @test_mm512_alignr_epi32
  // CHECK: @llvm.x86.avx512.mask.valign.d.512
  return _mm512_alignr_epi32(a, b, 2);
}

__m512i test_mm512_alignr_epi64(__m512i a, __m512i b)
{
  // CHECK-LABEL: @test_mm512_alignr_epi64
  // CHECK: @llvm.x86.avx512.mask.valign.q.512
  return _mm512_alignr_epi64(a, b, 2);
}

__m512d test_mm512_broadcastsd_pd(__m128d a)
{
  // CHECK-LABEL: @test_mm512_broadcastsd_pd
  // CHECK: insertelement <8 x double> {{.*}}, i32 0
  // CHECK: insertelement <8 x double> {{.*}}, i32 1
  // CHECK: insertelement <8 x double> {{.*}}, i32 2
  // CHECK: insertelement <8 x double> {{.*}}, i32 3
  // CHECK: insertelement <8 x double> {{.*}}, i32 4
  // CHECK: insertelement <8 x double> {{.*}}, i32 5
  // CHECK: insertelement <8 x double> {{.*}}, i32 6
  // CHECK: insertelement <8 x double> {{.*}}, i32 7
  return _mm512_broadcastsd_pd(a);
}

__m512d test_mm512_fmadd_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fmadd_round_pd(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}

__m512d test_mm512_mask_fmadd_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_mask_fmadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask3_fmadd_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.512
  return _mm512_mask3_fmadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_maskz_fmadd_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmadd_round_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fmadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_fmsub_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmsub_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fmsub_round_pd(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask_fmsub_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsub_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_mask_fmsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_maskz_fmsub_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsub_round_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fmsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_fnmadd_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fnmadd_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fnmadd_round_pd(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask3_fnmadd_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmadd_round_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.512
  return _mm512_mask3_fnmadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_maskz_fnmadd_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmadd_round_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fnmadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_fnmsub_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fnmsub_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fnmsub_round_pd(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_maskz_fnmsub_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmsub_round_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fnmsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_fmadd_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fmadd_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmadd_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_mask_fmadd_pd(__A, __U, __B, __C);
}
__m512d test_mm512_mask3_fmadd_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.512
  return _mm512_mask3_fmadd_pd(__A, __B, __C, __U);
}
__m512d test_mm512_maskz_fmadd_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fmadd_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fmsub_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fmsub_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmsub_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_mask_fmsub_pd(__A, __U, __B, __C);
}
__m512d test_mm512_maskz_fmsub_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fmsub_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fnmadd_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fnmadd_pd(__A, __B, __C);
}
__m512d test_mm512_mask3_fnmadd_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.pd.512
  return _mm512_mask3_fnmadd_pd(__A, __B, __C, __U);
}
__m512d test_mm512_maskz_fnmadd_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fnmadd_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fnmsub_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmadd.pd.512
  return _mm512_fnmsub_pd(__A, __B, __C);
}
__m512d test_mm512_maskz_fnmsub_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.pd.512
  return _mm512_maskz_fnmsub_pd(__U, __A, __B, __C);
}
__m512 test_mm512_fmadd_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fmadd_round_ps(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask_fmadd_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_mask_fmadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask3_fmadd_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.512
  return _mm512_mask3_fmadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_maskz_fmadd_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmadd_round_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fmadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_fmsub_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmsub_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fmsub_round_ps(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask_fmsub_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsub_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_mask_fmsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_maskz_fmsub_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsub_round_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fmsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_fnmadd_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fnmadd_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fnmadd_round_ps(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask3_fnmadd_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmadd_round_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.512
  return _mm512_mask3_fnmadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_maskz_fnmadd_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmadd_round_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fnmadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_fnmsub_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fnmsub_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fnmsub_round_ps(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_maskz_fnmsub_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmsub_round_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fnmsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_fmadd_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fmadd_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmadd_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_mask_fmadd_ps(__A, __U, __B, __C);
}
__m512 test_mm512_mask3_fmadd_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.512
  return _mm512_mask3_fmadd_ps(__A, __B, __C, __U);
}
__m512 test_mm512_maskz_fmadd_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fmadd_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fmsub_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fmsub_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmsub_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_mask_fmsub_ps(__A, __U, __B, __C);
}
__m512 test_mm512_maskz_fmsub_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fmsub_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fnmadd_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fnmadd_ps(__A, __B, __C);
}
__m512 test_mm512_mask3_fnmadd_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmadd.ps.512
  return _mm512_mask3_fnmadd_ps(__A, __B, __C, __U);
}
__m512 test_mm512_maskz_fnmadd_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fnmadd_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fnmsub_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmadd.ps.512
  return _mm512_fnmsub_ps(__A, __B, __C);
}
__m512 test_mm512_maskz_fnmsub_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fnmsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmadd.ps.512
  return _mm512_maskz_fnmsub_ps(__U, __A, __B, __C);
}
__m512d test_mm512_fmaddsub_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_fmaddsub_round_pd(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask_fmaddsub_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_mask_fmaddsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask3_fmaddsub_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.pd.512
  return _mm512_mask3_fmaddsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_maskz_fmaddsub_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmaddsub_round_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.512
  return _mm512_maskz_fmaddsub_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_fmsubadd_round_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmsubadd_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_fmsubadd_round_pd(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask_fmsubadd_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsubadd_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_mask_fmsubadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_maskz_fmsubadd_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsubadd_round_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.512
  return _mm512_maskz_fmsubadd_round_pd(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_fmaddsub_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_fmaddsub_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmaddsub_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_mask_fmaddsub_pd(__A, __U, __B, __C);
}
__m512d test_mm512_mask3_fmaddsub_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.pd.512
  return _mm512_mask3_fmaddsub_pd(__A, __B, __C, __U);
}
__m512d test_mm512_maskz_fmaddsub_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmaddsub_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.512
  return _mm512_maskz_fmaddsub_pd(__U, __A, __B, __C);
}
__m512d test_mm512_fmsubadd_pd(__m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_fmsubadd_pd(__A, __B, __C);
}
__m512d test_mm512_mask_fmsubadd_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.pd.512
  return _mm512_mask_fmsubadd_pd(__A, __U, __B, __C);
}
__m512d test_mm512_maskz_fmsubadd_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.pd.512
  return _mm512_maskz_fmsubadd_pd(__U, __A, __B, __C);
}
__m512 test_mm512_fmaddsub_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_fmaddsub_round_ps(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask_fmaddsub_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_mask_fmaddsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask3_fmaddsub_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.ps.512
  return _mm512_mask3_fmaddsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_maskz_fmaddsub_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmaddsub_round_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.512
  return _mm512_maskz_fmaddsub_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_fmsubadd_round_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmsubadd_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_fmsubadd_round_ps(__A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask_fmsubadd_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsubadd_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_mask_fmsubadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_maskz_fmsubadd_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsubadd_round_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.512
  return _mm512_maskz_fmsubadd_round_ps(__U, __A, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_fmaddsub_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_fmaddsub_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmaddsub_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_mask_fmaddsub_ps(__A, __U, __B, __C);
}
__m512 test_mm512_mask3_fmaddsub_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmaddsub.ps.512
  return _mm512_mask3_fmaddsub_ps(__A, __B, __C, __U);
}
__m512 test_mm512_maskz_fmaddsub_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmaddsub_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.512
  return _mm512_maskz_fmaddsub_ps(__U, __A, __B, __C);
}
__m512 test_mm512_fmsubadd_ps(__m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_fmsubadd_ps(__A, __B, __C);
}
__m512 test_mm512_mask_fmsubadd_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfmaddsub.ps.512
  return _mm512_mask_fmsubadd_ps(__A, __U, __B, __C);
}
__m512 test_mm512_maskz_fmsubadd_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_maskz_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.maskz.vfmaddsub.ps.512
  return _mm512_maskz_fmsubadd_ps(__U, __A, __B, __C);
}
__m512d test_mm512_mask3_fmsub_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsub_round_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.pd.512
  return _mm512_mask3_fmsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask3_fmsub_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.pd.512
  return _mm512_mask3_fmsub_pd(__A, __B, __C, __U);
}
__m512 test_mm512_mask3_fmsub_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsub_round_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.ps.512
  return _mm512_mask3_fmsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask3_fmsub_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsub.ps.512
  return _mm512_mask3_fmsub_ps(__A, __B, __C, __U);
}
__m512d test_mm512_mask3_fmsubadd_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsubadd_round_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.pd.512
  return _mm512_mask3_fmsubadd_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask3_fmsubadd_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsubadd_pd
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.pd.512
  return _mm512_mask3_fmsubadd_pd(__A, __B, __C, __U);
}
__m512 test_mm512_mask3_fmsubadd_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsubadd_round_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.ps.512
  return _mm512_mask3_fmsubadd_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask3_fmsubadd_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fmsubadd_ps
  // CHECK: @llvm.x86.avx512.mask3.vfmsubadd.ps.512
  return _mm512_mask3_fmsubadd_ps(__A, __B, __C, __U);
}
__m512d test_mm512_mask_fnmadd_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmadd_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.pd.512
  return _mm512_mask_fnmadd_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask_fnmadd_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmadd_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.pd.512
  return _mm512_mask_fnmadd_pd(__A, __U, __B, __C);
}
__m512 test_mm512_mask_fnmadd_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmadd_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.ps.512
  return _mm512_mask_fnmadd_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask_fnmadd_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmadd_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmadd.ps.512
  return _mm512_mask_fnmadd_ps(__A, __U, __B, __C);
}
__m512d test_mm512_mask_fnmsub_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmsub_round_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.pd.512
  return _mm512_mask_fnmsub_round_pd(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask3_fnmsub_round_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmsub_round_pd
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.pd.512
  return _mm512_mask3_fnmsub_round_pd(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512d test_mm512_mask_fnmsub_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512d __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.pd.512
  return _mm512_mask_fnmsub_pd(__A, __U, __B, __C);
}
__m512d test_mm512_mask3_fnmsub_pd(__m512d __A, __m512d __B, __m512d __C, __mmask8 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmsub_pd
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.pd.512
  return _mm512_mask3_fnmsub_pd(__A, __B, __C, __U);
}
__m512 test_mm512_mask_fnmsub_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmsub_round_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.ps.512
  return _mm512_mask_fnmsub_round_ps(__A, __U, __B, __C, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask3_fnmsub_round_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmsub_round_ps
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.ps.512
  return _mm512_mask3_fnmsub_round_ps(__A, __B, __C, __U, _MM_FROUND_TO_NEAREST_INT);
}
__m512 test_mm512_mask_fnmsub_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512 __C) {
  // CHECK-LABEL: @test_mm512_mask_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask.vfnmsub.ps.512
  return _mm512_mask_fnmsub_ps(__A, __U, __B, __C);
}
__m512 test_mm512_mask3_fnmsub_ps(__m512 __A, __m512 __B, __m512 __C, __mmask16 __U) {
  // CHECK-LABEL: @test_mm512_mask3_fnmsub_ps
  // CHECK: @llvm.x86.avx512.mask3.vfnmsub.ps.512
  return _mm512_mask3_fnmsub_ps(__A, __B, __C, __U);
}

__mmask16 test_mm512_cmpeq_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.512
  return (__mmask16)_mm512_cmpeq_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpeq_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.d.512
  return (__mmask16)_mm512_mask_cmpeq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_mask_cmpeq_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.512
  return (__mmask8)_mm512_mask_cmpeq_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpeq_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.q.512
  return (__mmask8)_mm512_cmpeq_epi64_mask(__a, __b);
}

__mmask16 test_mm512_cmpgt_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.512
  return (__mmask16)_mm512_cmpgt_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpgt_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.d.512
  return (__mmask16)_mm512_mask_cmpgt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_mask_cmpgt_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.512
  return (__mmask8)_mm512_mask_cmpgt_epi64_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpgt_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpgt.q.512
  return (__mmask8)_mm512_cmpgt_epi64_mask(__a, __b);
}

__m512d test_mm512_unpackhi_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_unpackhi_pd
  // CHECK: shufflevector <8 x double> {{.*}} <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  return _mm512_unpackhi_pd(a, b);
}

__m512d test_mm512_unpacklo_pd(__m512d a, __m512d b)
{
  // CHECK-LABEL: @test_mm512_unpacklo_pd
  // CHECK: shufflevector <8 x double> {{.*}} <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  return _mm512_unpacklo_pd(a, b);
}

__m512 test_mm512_unpackhi_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: @test_mm512_unpackhi_ps
  // CHECK: shufflevector <16 x float> {{.*}} <i32 2, i32 18, i32 3, i32 19, i32 6, i32 22, i32 7, i32 23, i32 10, i32 26, i32 11, i32 27, i32 14, i32 30, i32 15, i32 31>
  return _mm512_unpackhi_ps(a, b);
}

__m512 test_mm512_unpacklo_ps(__m512 a, __m512 b)
{
  // CHECK-LABEL: @test_mm512_unpacklo_ps
  // CHECK: shufflevector <16 x float> {{.*}} <i32 0, i32 16, i32 1, i32 17, i32 4, i32 20, i32 5, i32 21, i32 8, i32 24, i32 9, i32 25, i32 12, i32 28, i32 13, i32 29>
  return _mm512_unpacklo_ps(a, b);
}

__mmask16 test_mm512_cmp_round_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_cmp_round_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.512
  return _mm512_cmp_round_ps_mask(a, b, 0, _MM_FROUND_CUR_DIRECTION);
}

__mmask16 test_mm512_mask_cmp_round_ps_mask(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.512
  return _mm512_mask_cmp_round_ps_mask(m, a, b, 0, _MM_FROUND_CUR_DIRECTION);
}

__mmask16 test_mm512_cmp_ps_mask(__m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_cmp_ps_mask
  // CHECKn: @llvm.x86.avx512.mask.cmp.ps.512
  return _mm512_cmp_ps_mask(a, b, 0);
}

__mmask16 test_mm512_mask_cmp_ps_mask(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.512
  return _mm512_mask_cmp_ps_mask(m, a, b, 0);
}

__mmask8 test_mm512_cmp_round_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_cmp_round_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.512
  return _mm512_cmp_round_pd_mask(a, b, 0, _MM_FROUND_CUR_DIRECTION);
}

__mmask8 test_mm512_mask_cmp_round_pd_mask(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.512
  return _mm512_mask_cmp_round_pd_mask(m, a, b, 0, _MM_FROUND_CUR_DIRECTION);
}

__mmask8 test_mm512_cmp_pd_mask(__m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.512
  return _mm512_cmp_pd_mask(a, b, 0);
}

__mmask8 test_mm512_mask_cmp_pd_mask(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.512
  return _mm512_mask_cmp_pd_mask(m, a, b, 0);
}

__m256d test_mm512_extractf64x4_pd(__m512d a)
{
  // CHECK-LABEL: @test_mm512_extractf64x4_pd
  // CHECK: @llvm.x86.avx512.mask.vextractf64x4.512
  return _mm512_extractf64x4_pd(a, 1);
}

__m128 test_mm512_extractf32x4_ps(__m512 a)
{
  // CHECK-LABEL: @test_mm512_extractf32x4_ps
  // CHECK: @llvm.x86.avx512.mask.vextractf32x4.512
  return _mm512_extractf32x4_ps(a, 1);
}

__mmask16 test_mm512_cmpeq_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 0, i16 -1)
  return (__mmask16)_mm512_cmpeq_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpeq_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 0, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmpeq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpeq_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 0, i8 -1)
  return (__mmask8)_mm512_cmpeq_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpeq_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 0, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmpeq_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpge_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 5, i16 -1)
  return (__mmask16)_mm512_cmpge_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpge_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 5, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmpge_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpge_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm512_cmpge_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpge_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmpge_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpge_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 5, i16 -1)
  return (__mmask16)_mm512_cmpge_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpge_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 5, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmpge_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpge_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 5, i8 -1)
  return (__mmask8)_mm512_cmpge_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpge_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpge_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 5, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmpge_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpgt_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 6, i16 -1)
  return (__mmask16)_mm512_cmpgt_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpgt_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 6, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmpgt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpgt_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 6, i8 -1)
  return (__mmask8)_mm512_cmpgt_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpgt_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpgt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 6, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmpgt_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmple_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 2, i16 -1)
  return (__mmask16)_mm512_cmple_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmple_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 2, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmple_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmple_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm512_cmple_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmple_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmple_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmple_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 2, i16 -1)
  return (__mmask16)_mm512_cmple_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmple_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 2, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmple_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmple_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 2, i8 -1)
  return (__mmask8)_mm512_cmple_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmple_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmple_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 2, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmple_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmplt_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 1, i16 -1)
  return (__mmask16)_mm512_cmplt_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmplt_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 1, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmplt_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmplt_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm512_cmplt_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmplt_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmplt_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmplt_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 1, i16 -1)
  return (__mmask16)_mm512_cmplt_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmplt_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 1, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmplt_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmplt_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 1, i8 -1)
  return (__mmask8)_mm512_cmplt_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmplt_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmplt_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 1, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmplt_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpneq_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 4, i16 -1)
  return (__mmask16)_mm512_cmpneq_epi32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpneq_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 4, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmpneq_epi32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpneq_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm512_cmpneq_epi64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpneq_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmpneq_epi64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmpneq_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 4, i16 -1)
  return (__mmask16)_mm512_cmpneq_epu32_mask(__a, __b);
}

__mmask16 test_mm512_mask_cmpneq_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 4, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmpneq_epu32_mask(__u, __a, __b);
}

__mmask8 test_mm512_cmpneq_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 4, i8 -1)
  return (__mmask8)_mm512_cmpneq_epu64_mask(__a, __b);
}

__mmask8 test_mm512_mask_cmpneq_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpneq_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 4, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmpneq_epu64_mask(__u, __a, __b);
}

__mmask16 test_mm512_cmp_epi32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 3, i16 -1)
  return (__mmask16)_mm512_cmp_epi32_mask(__a, __b, 3);
}

__mmask16 test_mm512_mask_cmp_epi32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epi32_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 3, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmp_epi32_mask(__u, __a, __b, 3);
}

__mmask8 test_mm512_cmp_epi64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 3, i8 -1)
  return (__mmask8)_mm512_cmp_epi64_mask(__a, __b, 3);
}

__mmask8 test_mm512_mask_cmp_epi64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epi64_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 3, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmp_epi64_mask(__u, __a, __b, 3);
}

__mmask16 test_mm512_cmp_epu32_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 3, i16 -1)
  return (__mmask16)_mm512_cmp_epu32_mask(__a, __b, 3);
}

__mmask16 test_mm512_mask_cmp_epu32_mask(__mmask16 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epu32_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.d.512(<16 x i32> {{.*}}, <16 x i32> {{.*}}, i32 3, i16 {{.*}})
  return (__mmask16)_mm512_mask_cmp_epu32_mask(__u, __a, __b, 3);
}

__mmask8 test_mm512_cmp_epu64_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 3, i8 -1)
  return (__mmask8)_mm512_cmp_epu64_mask(__a, __b, 3);
}

__mmask8 test_mm512_mask_cmp_epu64_mask(__mmask8 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_epu64_mask
  // CHECK: @llvm.x86.avx512.mask.ucmp.q.512(<8 x i64> {{.*}}, <8 x i64> {{.*}}, i32 3, i8 {{.*}})
  return (__mmask8)_mm512_mask_cmp_epu64_mask(__u, __a, __b, 3);
}

__m512i test_mm512_mask_and_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_and_epi32
  // CHECK: @llvm.x86.avx512.mask.pand.d.512
  return _mm512_mask_and_epi32(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_and_epi32(__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_maskz_and_epi32
  // CHECK: @llvm.x86.avx512.mask.pand.d.512
  return _mm512_maskz_and_epi32(__k,__a, __b);
}

__m512i test_mm512_mask_and_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_and_epi64
  // CHECK: @llvm.x86.avx512.mask.pand.q.512
  return _mm512_mask_and_epi64(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_and_epi64(__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_maskz_and_epi64
  // CHECK: @llvm.x86.avx512.mask.pand.q.512
  return _mm512_maskz_and_epi64(__k,__a, __b);
}

__m512i test_mm512_mask_or_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_or_epi32
  // CHECK: @llvm.x86.avx512.mask.por.d.512
  return _mm512_mask_or_epi32(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_or_epi32(__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_maskz_or_epi32
  // CHECK: @llvm.x86.avx512.mask.por.d.512
  return _mm512_maskz_or_epi32(__k,__a, __b);
}

__m512i test_mm512_mask_or_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_or_epi64
  // CHECK: @llvm.x86.avx512.mask.por.q.512
  return _mm512_mask_or_epi64(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_or_epi64(__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_maskz_or_epi64
  // CHECK: @llvm.x86.avx512.mask.por.q.512
  return _mm512_maskz_or_epi64(__k,__a, __b);
}

__m512i test_mm512_mask_xor_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_xor_epi32
  // CHECK: @llvm.x86.avx512.mask.pxor.d.512
  return _mm512_mask_xor_epi32(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_xor_epi32(__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_maskz_xor_epi32
  // CHECK: @llvm.x86.avx512.mask.pxor.d.512
  return _mm512_maskz_xor_epi32(__k,__a, __b);
}

__m512i test_mm512_mask_xor_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_xor_epi64
  // CHECK: @llvm.x86.avx512.mask.pxor.q.512
  return _mm512_mask_xor_epi64(__src, __k,__a, __b);
}

__m512i test_mm512_maskz_xor_epi64(__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_maskz_xor_epi64
  // CHECK: @llvm.x86.avx512.mask.pxor.q.512
  return _mm512_maskz_xor_epi64(__k,__a, __b);
}

__m512i test_mm512_and_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_and_epi32
  // CHECK: and <8 x i64>
  return _mm512_and_epi32(__a, __b);
}

__m512i test_mm512_and_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_and_epi64
  // CHECK: and <8 x i64>
  return _mm512_and_epi64(__a, __b);
}

__m512i test_mm512_or_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_or_epi32
  // CHECK: or <8 x i64>
  return _mm512_or_epi32(__a, __b);
}

__m512i test_mm512_or_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_or_epi64
  // CHECK: or <8 x i64>
  return _mm512_or_epi64(__a, __b);
}

__m512i test_mm512_xor_epi32(__m512i __src,__mmask16 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_xor_epi32
  // CHECK: xor <8 x i64>
  return _mm512_xor_epi32(__a, __b);
}

__m512i test_mm512_xor_epi64(__m512i __src,__mmask8 __k, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_xor_epi64
  // CHECK: xor <8 x i64>
  return _mm512_xor_epi64(__a, __b);
}

__m512i test_mm512_maskz_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B){
  //CHECK-LABEL: @test_mm512_maskz_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_maskz_andnot_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi32 (__mmask16 __k,__m512i __A, __m512i __B,
                                      __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_mask_andnot_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_andnot_epi32
  //CHECK: @llvm.x86.avx512.mask.pandn.d.512
  return _mm512_andnot_epi32(__A,__B);
}

__m512i test_mm512_maskz_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_maskz_andnot_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_andnot_epi64 (__mmask8 __k,__m512i __A, __m512i __B, 
                                      __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_mask_andnot_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_andnot_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_andnot_epi64
  //CHECK: @llvm.x86.avx512.mask.pandn.q.512
  return _mm512_andnot_epi64(__A,__B);
}

__m512i test_mm512_maskz_sub_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.512
  return _mm512_maskz_sub_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_sub_epi32 (__mmask16 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_sub_epi32
  //CHECK: @llvm.x86.avx512.mask.psub.d.512
  return _mm512_mask_sub_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_sub_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_sub_epi32
  //CHECK: sub <16 x i32>
  return _mm512_sub_epi32(__A,__B);
}

__m512i test_mm512_maskz_sub_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.512
  return _mm512_maskz_sub_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_sub_epi64 (__mmask8 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_sub_epi64
  //CHECK: @llvm.x86.avx512.mask.psub.q.512
  return _mm512_mask_sub_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_sub_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_sub_epi64
  //CHECK: sub <8 x i64>
  return _mm512_sub_epi64(__A,__B);
}

__m512i test_mm512_maskz_add_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.512
  return _mm512_maskz_add_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_add_epi32 (__mmask16 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_add_epi32
  //CHECK: @llvm.x86.avx512.mask.padd.d.512
  return _mm512_mask_add_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_add_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_add_epi32
  //CHECK: add <16 x i32>
  return _mm512_add_epi32(__A,__B);
}

__m512i test_mm512_maskz_add_epi64 (__mmask8 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.512
  return _mm512_maskz_add_epi64(__k,__A,__B);
}

__m512i test_mm512_mask_add_epi64 (__mmask8 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_add_epi64
  //CHECK: @llvm.x86.avx512.mask.padd.q.512
  return _mm512_mask_add_epi64(__src,__k,__A,__B);
}

__m512i test_mm512_add_epi64(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_add_epi64
  //CHECK: add <8 x i64>
  return _mm512_add_epi64(__A,__B);
}

__m512i test_mm512_maskz_mul_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.512
  return _mm512_maskz_mul_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_mul_epi32 (__mmask16 __k,__m512i __A, __m512i __B,
                                   __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_mul_epi32
  //CHECK: @llvm.x86.avx512.mask.pmul.dq.512
  return _mm512_mask_mul_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_maskz_mul_epu32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.512
  return _mm512_maskz_mul_epu32(__k,__A,__B);
}

__m512i test_mm512_mask_mul_epu32 (__mmask16 __k,__m512i __A, __m512i __B, 
                                   __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_mul_epu32
  //CHECK: @llvm.x86.avx512.mask.pmulu.dq.512
  return _mm512_mask_mul_epu32(__src,__k,__A,__B);
}

__m512i test_mm512_maskz_mullo_epi32 (__mmask16 __k,__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_maskz_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.512
  return _mm512_maskz_mullo_epi32(__k,__A,__B);
}

__m512i test_mm512_mask_mullo_epi32 (__mmask16 __k,__m512i __A, __m512i __B, __m512i __src) {
  //CHECK-LABEL: @test_mm512_mask_mullo_epi32
  //CHECK: @llvm.x86.avx512.mask.pmull.d.512
  return _mm512_mask_mullo_epi32(__src,__k,__A,__B);
}

__m512i test_mm512_mullo_epi32(__m512i __A, __m512i __B) {
  //CHECK-LABEL: @test_mm512_mullo_epi32
  //CHECK: mul <16 x i32>
  return _mm512_mullo_epi32(__A,__B);
}

__m512d test_mm512_add_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_add_round_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.512
  return _mm512_add_round_pd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_add_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_add_round_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.512
  return _mm512_mask_add_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_maskz_add_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_add_round_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.512
  return _mm512_maskz_add_round_pd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_add_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_add_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.512
  return _mm512_mask_add_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_add_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_add_pd
  // CHECK: @llvm.x86.avx512.mask.add.pd.512
  return _mm512_maskz_add_pd(__U,__A,__B); 
}
__m512 test_mm512_add_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_add_round_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.512
  return _mm512_add_round_ps(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_add_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_add_round_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.512
  return _mm512_mask_add_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_maskz_add_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_add_round_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.512
  return _mm512_maskz_add_round_ps(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_add_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_add_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.512
  return _mm512_mask_add_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_add_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_add_ps
  // CHECK: @llvm.x86.avx512.mask.add.ps.512
  return _mm512_maskz_add_ps(__U,__A,__B); 
}
__m128 test_mm_add_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_add_round_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_add_round_ss(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_add_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_add_round_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_mask_add_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_maskz_add_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_add_round_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_maskz_add_round_ss(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_add_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_add_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_mask_add_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_add_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_add_ss
  // CHECK: @llvm.x86.avx512.mask.add.ss.round
  return _mm_maskz_add_ss(__U,__A,__B); 
}
__m128d test_mm_add_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_add_round_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_add_round_sd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_add_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_add_round_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_mask_add_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_maskz_add_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_add_round_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_maskz_add_round_sd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_add_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_add_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_mask_add_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_add_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_add_sd
  // CHECK: @llvm.x86.avx512.mask.add.sd.round
  return _mm_maskz_add_sd(__U,__A,__B); 
}
__m512d test_mm512_sub_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_sub_round_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.512
  return _mm512_sub_round_pd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_sub_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_sub_round_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.512
  return _mm512_mask_sub_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_maskz_sub_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_sub_round_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.512
  return _mm512_maskz_sub_round_pd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_sub_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_sub_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.512
  return _mm512_mask_sub_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_sub_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_sub_pd
  // CHECK: @llvm.x86.avx512.mask.sub.pd.512
  return _mm512_maskz_sub_pd(__U,__A,__B); 
}
__m512 test_mm512_sub_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_sub_round_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.512
  return _mm512_sub_round_ps(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_sub_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_sub_round_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.512
  return _mm512_mask_sub_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_maskz_sub_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_sub_round_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.512
  return _mm512_maskz_sub_round_ps(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_sub_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_sub_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.512
  return _mm512_mask_sub_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_sub_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_sub_ps
  // CHECK: @llvm.x86.avx512.mask.sub.ps.512
  return _mm512_maskz_sub_ps(__U,__A,__B); 
}
__m128 test_mm_sub_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_sub_round_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_sub_round_ss(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_sub_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_sub_round_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_mask_sub_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_maskz_sub_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_round_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_maskz_sub_round_ss(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_sub_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_sub_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_mask_sub_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_sub_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_ss
  // CHECK: @llvm.x86.avx512.mask.sub.ss.round
  return _mm_maskz_sub_ss(__U,__A,__B); 
}
__m128d test_mm_sub_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_sub_round_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_sub_round_sd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_sub_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_sub_round_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_mask_sub_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_maskz_sub_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_round_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_maskz_sub_round_sd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_sub_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_sub_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_mask_sub_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_sub_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_sub_sd
  // CHECK: @llvm.x86.avx512.mask.sub.sd.round
  return _mm_maskz_sub_sd(__U,__A,__B); 
}
__m512d test_mm512_mul_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mul_round_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.512
  return _mm512_mul_round_pd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_mul_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_mul_round_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.512
  return _mm512_mask_mul_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_maskz_mul_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_mul_round_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.512
  return _mm512_maskz_mul_round_pd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_mul_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_mul_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.512
  return _mm512_mask_mul_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_mul_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_mul_pd
  // CHECK: @llvm.x86.avx512.mask.mul.pd.512
  return _mm512_maskz_mul_pd(__U,__A,__B); 
}
__m512 test_mm512_mul_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mul_round_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.512
  return _mm512_mul_round_ps(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_mul_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_mul_round_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.512
  return _mm512_mask_mul_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_maskz_mul_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_mul_round_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.512
  return _mm512_maskz_mul_round_ps(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_mul_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_mul_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.512
  return _mm512_mask_mul_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_mul_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_mul_ps
  // CHECK: @llvm.x86.avx512.mask.mul.ps.512
  return _mm512_maskz_mul_ps(__U,__A,__B); 
}
__m128 test_mm_mul_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mul_round_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_mul_round_ss(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_mul_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_mul_round_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_mask_mul_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_maskz_mul_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_round_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_maskz_mul_round_ss(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_mul_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_mul_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_mask_mul_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_mul_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_ss
  // CHECK: @llvm.x86.avx512.mask.mul.ss.round
  return _mm_maskz_mul_ss(__U,__A,__B); 
}
__m128d test_mm_mul_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mul_round_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_mul_round_sd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_mul_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_mul_round_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_mask_mul_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_maskz_mul_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_round_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_maskz_mul_round_sd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_mul_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_mul_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_mask_mul_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_mul_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_mul_sd
  // CHECK: @llvm.x86.avx512.mask.mul.sd.round
  return _mm_maskz_mul_sd(__U,__A,__B); 
}
__m512d test_mm512_div_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_div_round_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.512
  return _mm512_div_round_pd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_div_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_div_round_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.512
  return _mm512_mask_div_round_pd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_maskz_div_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_div_round_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.512
  return _mm512_maskz_div_round_pd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512d test_mm512_mask_div_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_div_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.512
  return _mm512_mask_div_pd(__W,__U,__A,__B); 
}
__m512d test_mm512_maskz_div_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_div_pd
  // CHECK: @llvm.x86.avx512.mask.div.pd.512
  return _mm512_maskz_div_pd(__U,__A,__B); 
}
__m512 test_mm512_div_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_div_round_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.512
  return _mm512_div_round_ps(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_div_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_div_round_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.512
  return _mm512_mask_div_round_ps(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_maskz_div_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_div_round_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.512
  return _mm512_maskz_div_round_ps(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m512 test_mm512_mask_div_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_div_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.512
  return _mm512_mask_div_ps(__W,__U,__A,__B); 
}
__m512 test_mm512_maskz_div_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_div_ps
  // CHECK: @llvm.x86.avx512.mask.div.ps.512
  return _mm512_maskz_div_ps(__U,__A,__B); 
}
__m128 test_mm_div_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_div_round_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_div_round_ss(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_div_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_div_round_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_mask_div_round_ss(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_maskz_div_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_div_round_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_maskz_div_round_ss(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128 test_mm_mask_div_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_div_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_mask_div_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_div_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_div_ss
  // CHECK: @llvm.x86.avx512.mask.div.ss.round
  return _mm_maskz_div_ss(__U,__A,__B); 
}
__m128d test_mm_div_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_div_round_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_div_round_sd(__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_div_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_div_round_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_mask_div_round_sd(__W,__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_maskz_div_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_div_round_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_maskz_div_round_sd(__U,__A,__B,_MM_FROUND_TO_NEAREST_INT); 
}
__m128d test_mm_mask_div_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_div_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_mask_div_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_div_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_div_sd
  // CHECK: @llvm.x86.avx512.mask.div.sd.round
  return _mm_maskz_div_sd(__U,__A,__B); 
}
__m128 test_mm_max_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_max_round_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_max_round_ss(__A,__B,0x08); 
}
__m128 test_mm_mask_max_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_max_round_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_mask_max_round_ss(__W,__U,__A,__B,0x08); 
}
__m128 test_mm_maskz_max_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_max_round_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_maskz_max_round_ss(__U,__A,__B,0x08); 
}
__m128 test_mm_mask_max_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_max_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_mask_max_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_max_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_max_ss
  // CHECK: @llvm.x86.avx512.mask.max.ss.round
  return _mm_maskz_max_ss(__U,__A,__B); 
}
__m128d test_mm_max_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_max_round_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_max_round_sd(__A,__B,0x08); 
}
__m128d test_mm_mask_max_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_max_round_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_mask_max_round_sd(__W,__U,__A,__B,0x08); 
}
__m128d test_mm_maskz_max_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_max_round_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_maskz_max_round_sd(__U,__A,__B,0x08); 
}
__m128d test_mm_mask_max_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_max_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_mask_max_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_max_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_max_sd
  // CHECK: @llvm.x86.avx512.mask.max.sd.round
  return _mm_maskz_max_sd(__U,__A,__B); 
}
__m128 test_mm_min_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_min_round_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_min_round_ss(__A,__B,0x08); 
}
__m128 test_mm_mask_min_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_min_round_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_mask_min_round_ss(__W,__U,__A,__B,0x08); 
}
__m128 test_mm_maskz_min_round_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_min_round_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_maskz_min_round_ss(__U,__A,__B,0x08); 
}
__m128 test_mm_mask_min_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_min_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_mask_min_ss(__W,__U,__A,__B); 
}
__m128 test_mm_maskz_min_ss(__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_min_ss
  // CHECK: @llvm.x86.avx512.mask.min.ss.round
  return _mm_maskz_min_ss(__U,__A,__B); 
}
__m128d test_mm_min_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_min_round_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_min_round_sd(__A,__B,0x08); 
}
__m128d test_mm_mask_min_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_min_round_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_mask_min_round_sd(__W,__U,__A,__B,0x08); 
}
__m128d test_mm_maskz_min_round_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_min_round_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_maskz_min_round_sd(__U,__A,__B,0x08); 
}
__m128d test_mm_mask_min_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_min_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_mask_min_sd(__W,__U,__A,__B); 
}
__m128d test_mm_maskz_min_sd(__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_min_sd
  // CHECK: @llvm.x86.avx512.mask.min.sd.round
  return _mm_maskz_min_sd(__U,__A,__B); 
}

__m512 test_mm512_undefined() {
  // CHECK-LABEL: @test_mm512_undefined
  // CHECK: ret <16 x float> undef
  return _mm512_undefined();
}

__m512 test_mm512_undefined_ps() {
  // CHECK-LABEL: @test_mm512_undefined_ps
  // CHECK: ret <16 x float> undef
  return _mm512_undefined_ps();
}

__m512d test_mm512_undefined_pd() {
  // CHECK-LABEL: @test_mm512_undefined_pd
  // CHECK: ret <8 x double> undef
  return _mm512_undefined_pd();
}

__m512i test_mm512_undefined_epi32() {
  // CHECK-LABEL: @test_mm512_undefined_epi32
  // CHECK: ret <8 x i64> undef
  return _mm512_undefined_epi32();
}

__m512i test_mm512_cvtepi8_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.512
  return _mm512_cvtepi8_epi32(__A); 
}

__m512i test_mm512_mask_cvtepi8_epi32(__m512i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.512
  return _mm512_mask_cvtepi8_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi8_epi32(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.d.512
  return _mm512_maskz_cvtepi8_epi32(__U, __A); 
}

__m512i test_mm512_cvtepi8_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.512
  return _mm512_cvtepi8_epi64(__A); 
}

__m512i test_mm512_mask_cvtepi8_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.512
  return _mm512_mask_cvtepi8_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxb.q.512
  return _mm512_maskz_cvtepi8_epi64(__U, __A); 
}

__m512i test_mm512_cvtepi32_epi64(__m256i __X) {
  // CHECK-LABEL: @test_mm512_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.512
  return _mm512_cvtepi32_epi64(__X); 
}

__m512i test_mm512_mask_cvtepi32_epi64(__m512i __W, __mmask8 __U, __m256i __X) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.512
  return _mm512_mask_cvtepi32_epi64(__W, __U, __X); 
}

__m512i test_mm512_maskz_cvtepi32_epi64(__mmask8 __U, __m256i __X) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxd.q.512
  return _mm512_maskz_cvtepi32_epi64(__U, __X); 
}

__m512i test_mm512_cvtepi16_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.512
  return _mm512_cvtepi16_epi32(__A); 
}

__m512i test_mm512_mask_cvtepi16_epi32(__m512i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.512
  return _mm512_mask_cvtepi16_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi16_epi32(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.d.512
  return _mm512_maskz_cvtepi16_epi32(__U, __A); 
}

__m512i test_mm512_cvtepi16_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.512
  return _mm512_cvtepi16_epi64(__A); 
}

__m512i test_mm512_mask_cvtepi16_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.512
  return _mm512_mask_cvtepi16_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepi16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovsxw.q.512
  return _mm512_maskz_cvtepi16_epi64(__U, __A); 
}

__m512i test_mm512_cvtepu8_epi32(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.512
  return _mm512_cvtepu8_epi32(__A); 
}

__m512i test_mm512_mask_cvtepu8_epi32(__m512i __W, __mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.512
  return _mm512_mask_cvtepu8_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu8_epi32(__mmask16 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu8_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.d.512
  return _mm512_maskz_cvtepu8_epi32(__U, __A); 
}

__m512i test_mm512_cvtepu8_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.512
  return _mm512_cvtepu8_epi64(__A); 
}

__m512i test_mm512_mask_cvtepu8_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.512
  return _mm512_mask_cvtepu8_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu8_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu8_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxb.q.512
  return _mm512_maskz_cvtepu8_epi64(__U, __A); 
}

__m512i test_mm512_cvtepu32_epi64(__m256i __X) {
  // CHECK-LABEL: @test_mm512_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.512
  return _mm512_cvtepu32_epi64(__X); 
}

__m512i test_mm512_mask_cvtepu32_epi64(__m512i __W, __mmask8 __U, __m256i __X) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.512
  return _mm512_mask_cvtepu32_epi64(__W, __U, __X); 
}

__m512i test_mm512_maskz_cvtepu32_epi64(__mmask8 __U, __m256i __X) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu32_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxd.q.512
  return _mm512_maskz_cvtepu32_epi64(__U, __X); 
}

__m512i test_mm512_cvtepu16_epi32(__m256i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.512
  return _mm512_cvtepu16_epi32(__A); 
}

__m512i test_mm512_mask_cvtepu16_epi32(__m512i __W, __mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.512
  return _mm512_mask_cvtepu16_epi32(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu16_epi32(__mmask16 __U, __m256i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu16_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.d.512
  return _mm512_maskz_cvtepu16_epi32(__U, __A); 
}

__m512i test_mm512_cvtepu16_epi64(__m128i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.512
  return _mm512_cvtepu16_epi64(__A); 
}

__m512i test_mm512_mask_cvtepu16_epi64(__m512i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.512
  return _mm512_mask_cvtepu16_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtepu16_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu16_epi64
  // CHECK: @llvm.x86.avx512.mask.pmovzxw.q.512
  return _mm512_maskz_cvtepu16_epi64(__U, __A); 
}


__m512i test_mm512_rol_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.512
  return _mm512_rol_epi32(__A, 5); 
}

__m512i test_mm512_mask_rol_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.512
  return _mm512_mask_rol_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_rol_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_rol_epi32
  // CHECK: @llvm.x86.avx512.mask.prol.d.512
  return _mm512_maskz_rol_epi32(__U, __A, 5); 
}

__m512i test_mm512_rol_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.512
  return _mm512_rol_epi64(__A, 5); 
}

__m512i test_mm512_mask_rol_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.512
  return _mm512_mask_rol_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_rol_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_rol_epi64
  // CHECK: @llvm.x86.avx512.mask.prol.q.512
  return _mm512_maskz_rol_epi64(__U, __A, 5); 
}

__m512i test_mm512_rolv_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.512
  return _mm512_rolv_epi32(__A, __B); 
}

__m512i test_mm512_mask_rolv_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.512
  return _mm512_mask_rolv_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rolv_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_rolv_epi32
  // CHECK: @llvm.x86.avx512.mask.prolv.d.512
  return _mm512_maskz_rolv_epi32(__U, __A, __B); 
}

__m512i test_mm512_rolv_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.512
  return _mm512_rolv_epi64(__A, __B); 
}

__m512i test_mm512_mask_rolv_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.512
  return _mm512_mask_rolv_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rolv_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_rolv_epi64
  // CHECK: @llvm.x86.avx512.mask.prolv.q.512
  return _mm512_maskz_rolv_epi64(__U, __A, __B); 
}

__m512i test_mm512_ror_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.512
  return _mm512_ror_epi32(__A, 5); 
}

__m512i test_mm512_mask_ror_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.512
  return _mm512_mask_ror_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_ror_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_ror_epi32
  // CHECK: @llvm.x86.avx512.mask.pror.d.512
  return _mm512_maskz_ror_epi32(__U, __A, 5); 
}

__m512i test_mm512_ror_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.512
  return _mm512_ror_epi64(__A, 5); 
}

__m512i test_mm512_mask_ror_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.512
  return _mm512_mask_ror_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_ror_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_ror_epi64
  // CHECK: @llvm.x86.avx512.mask.pror.q.512
  return _mm512_maskz_ror_epi64(__U, __A, 5); 
}


__m512i test_mm512_rorv_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.512
  return _mm512_rorv_epi32(__A, __B); 
}

__m512i test_mm512_mask_rorv_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.512
  return _mm512_mask_rorv_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rorv_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_rorv_epi32
  // CHECK: @llvm.x86.avx512.mask.prorv.d.512
  return _mm512_maskz_rorv_epi32(__U, __A, __B); 
}

__m512i test_mm512_rorv_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.512
  return _mm512_rorv_epi64(__A, __B); 
}

__m512i test_mm512_mask_rorv_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.512
  return _mm512_mask_rorv_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_rorv_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_rorv_epi64
  // CHECK: @llvm.x86.avx512.mask.prorv.q.512
  return _mm512_maskz_rorv_epi64(__U, __A, __B); 
}

__m512i test_mm512_slli_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_slli_epi32
  // CHECK: @llvm.x86.avx512.mask.psll.di.512
  return _mm512_slli_epi32(__A, 5); 
}

__m512i test_mm512_mask_slli_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_slli_epi32
  // CHECK: @llvm.x86.avx512.mask.psll.di.512
  return _mm512_mask_slli_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_slli_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_slli_epi32
  // CHECK: @llvm.x86.avx512.mask.psll.di.512
  return _mm512_maskz_slli_epi32(__U, __A, 5); 
}

__m512i test_mm512_slli_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_slli_epi64
  // CHECK: @llvm.x86.avx512.mask.psll.qi.512
  return _mm512_slli_epi64(__A, 5); 
}

__m512i test_mm512_mask_slli_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_slli_epi64
  // CHECK: @llvm.x86.avx512.mask.psll.qi.512
  return _mm512_mask_slli_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_slli_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_slli_epi64
  // CHECK: @llvm.x86.avx512.mask.psll.qi.512
  return _mm512_maskz_slli_epi64(__U, __A, 5); 
}

__m512i test_mm512_srli_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.512
  return _mm512_srli_epi32(__A, 5); 
}

__m512i test_mm512_mask_srli_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.512
  return _mm512_mask_srli_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_srli_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_srli_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.di.512
  return _mm512_maskz_srli_epi32(__U, __A, 5); 
}

__m512i test_mm512_srli_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.512
  return _mm512_srli_epi64(__A, 5); 
}

__m512i test_mm512_mask_srli_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.512
  return _mm512_mask_srli_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_srli_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_srli_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.qi.512
  return _mm512_maskz_srli_epi64(__U, __A, 5); 
}

__m512i test_mm512_mask_load_epi32(__m512i __W, __mmask16 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_mask_load_epi32
  // CHECK: @llvm.x86.avx512.mask.load.d.512
  return _mm512_mask_load_epi32(__W, __U, __P); 
}

__m512i test_mm512_maskz_load_epi32(__mmask16 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_maskz_load_epi32
  // CHECK: @llvm.x86.avx512.mask.load.d.512
  return _mm512_maskz_load_epi32(__U, __P); 
}

__m512i test_mm512_mask_mov_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_mov_epi64
  // CHECK: @llvm.x86.avx512.mask.mov
  return _mm512_mask_mov_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_mov_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_mov_epi64
  // CHECK: @llvm.x86.avx512.mask.mov
  return _mm512_maskz_mov_epi64(__U, __A); 
}

__m512i test_mm512_mask_load_epi64(__m512i __W, __mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_mask_load_epi64
  // CHECK: @llvm.x86.avx512.mask.load.q.512
  return _mm512_mask_load_epi64(__W, __U, __P); 
}

__m512i test_mm512_maskz_load_epi64(__mmask8 __U, void const *__P) {
  // CHECK-LABEL: @test_mm512_maskz_load_epi64
  // CHECK: @llvm.x86.avx512.mask.load.q.512
  return _mm512_maskz_load_epi64(__U, __P); 
}

void test_mm512_mask_store_epi32(void *__P, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_store_epi32
  // CHECK: @llvm.x86.avx512.mask.store.d.512
  return _mm512_mask_store_epi32(__P, __U, __A); 
}

void test_mm512_mask_store_epi64(void *__P, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_store_epi64
  // CHECK: @llvm.x86.avx512.mask.store.q.512
  return _mm512_mask_store_epi64(__P, __U, __A); 
}

__m512d test_mm512_movedup_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_movedup_pd
  // CHECK: @llvm.x86.avx512.mask.movddup.512
  return _mm512_movedup_pd(__A); 
}

__m512d test_mm512_mask_movedup_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_movedup_pd
  // CHECK: @llvm.x86.avx512.mask.movddup.512
  return _mm512_mask_movedup_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_movedup_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_movedup_pd
  // CHECK: @llvm.x86.avx512.mask.movddup.512
  return _mm512_maskz_movedup_pd(__U, __A); 
}

int test_mm_comi_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_comi_round_sd
  // CHECK: @llvm.x86.avx512.vcomi.sd
  return _mm_comi_round_sd(__A, __B, 5, 3); 
}

int test_mm_comi_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_comi_round_ss
  // CHECK: @llvm.x86.avx512.vcomi.ss
  return _mm_comi_round_ss(__A, __B, 5, 3); 
}

__m512d test_mm512_fixupimm_round_pd(__m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_fixupimm_round_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_fixupimm_round_pd(__A, __B, __C, 5, 8); 
}

__m512d test_mm512_mask_fixupimm_round_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_fixupimm_round_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_mask_fixupimm_round_pd(__A, __U, __B, __C, 5, 8); 
}

__m512d test_mm512_fixupimm_pd(__m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_fixupimm_pd(__A, __B, __C, 5); 
}

__m512d test_mm512_mask_fixupimm_pd(__m512d __A, __mmask8 __U, __m512d __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_fixupimm_pd
  // CHECK: @llvm.x86.avx512.mask.fixupimm.pd.512
  return _mm512_mask_fixupimm_pd(__A, __U, __B, __C, 5); 
}

__m512d test_mm512_maskz_fixupimm_round_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_fixupimm_round_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.512
  return _mm512_maskz_fixupimm_round_pd(__U, __A, __B, __C, 5, 8); 
}

__m512d test_mm512_maskz_fixupimm_pd(__mmask8 __U, __m512d __A, __m512d __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_fixupimm_pd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.pd.512
  return _mm512_maskz_fixupimm_pd(__U, __A, __B, __C, 5); 
}

__m512 test_mm512_fixupimm_round_ps(__m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_fixupimm_round_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_fixupimm_round_ps(__A, __B, __C, 5, 8); 
}

__m512 test_mm512_mask_fixupimm_round_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_fixupimm_round_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_mask_fixupimm_round_ps(__A, __U, __B, __C, 5, 8); 
}

__m512 test_mm512_fixupimm_ps(__m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_fixupimm_ps(__A, __B, __C, 5); 
}

__m512 test_mm512_mask_fixupimm_ps(__m512 __A, __mmask16 __U, __m512 __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_fixupimm_ps
  // CHECK: @llvm.x86.avx512.mask.fixupimm.ps.512
  return _mm512_mask_fixupimm_ps(__A, __U, __B, __C, 5); 
}

__m512 test_mm512_maskz_fixupimm_round_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_fixupimm_round_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.512
  return _mm512_maskz_fixupimm_round_ps(__U, __A, __B, __C, 5, 8); 
}

__m512 test_mm512_maskz_fixupimm_ps(__mmask16 __U, __m512 __A, __m512 __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_fixupimm_ps
  // CHECK: @llvm.x86.avx512.maskz.fixupimm.ps.512
  return _mm512_maskz_fixupimm_ps(__U, __A, __B, __C, 5); 
}

__m128d test_mm_fixupimm_round_sd(__m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_round_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_round_sd(__A, __B, __C, 5, 8); 
}

__m128d test_mm_mask_fixupimm_round_sd(__m128d __A, __mmask8 __U, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_round_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_round_sd(__A, __U, __B, __C, 5, 8); 
}

__m128d test_mm_fixupimm_sd(__m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_sd(__A, __B, __C, 5); 
}

__m128d test_mm_mask_fixupimm_sd(__m128d __A, __mmask8 __U, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_sd
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_sd(__A, __U, __B, __C, 5); 
}

__m128d test_mm_maskz_fixupimm_round_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_round_sd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_round_sd(__U, __A, __B, __C, 5, 8); 
}

__m128d test_mm_maskz_fixupimm_sd(__mmask8 __U, __m128d __A, __m128d __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_sd
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_sd(__U, __A, __B, __C, 5); 
}

__m128 test_mm_fixupimm_round_ss(__m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_round_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_round_ss(__A, __B, __C, 5, 8); 
}

__m128 test_mm_mask_fixupimm_round_ss(__m128 __A, __mmask8 __U, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_round_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_round_ss(__A, __U, __B, __C, 5, 8); 
}

__m128 test_mm_fixupimm_ss(__m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_fixupimm_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_fixupimm_ss(__A, __B, __C, 5); 
}

__m128 test_mm_mask_fixupimm_ss(__m128 __A, __mmask8 __U, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_mask_fixupimm_ss
  // CHECK: @llvm.x86.avx512.mask.fixupimm
  return _mm_mask_fixupimm_ss(__A, __U, __B, __C, 5); 
}

__m128 test_mm_maskz_fixupimm_round_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_round_ss
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_round_ss(__U, __A, __B, __C, 5, 8); 
}

__m128 test_mm_maskz_fixupimm_ss(__mmask8 __U, __m128 __A, __m128 __B, __m128i __C) {
  // CHECK-LABEL: @test_mm_maskz_fixupimm_ss
  // CHECK: @llvm.x86.avx512.maskz.fixupimm
  return _mm_maskz_fixupimm_ss(__U, __A, __B, __C, 5); 
}

__m128d test_mm_getexp_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_getexp_round_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_getexp_round_sd(__A, __B, 8); 
}

__m128d test_mm_getexp_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_getexp_sd
  // CHECK: @llvm.x86.avx512.mask.getexp.sd
  return _mm_getexp_sd(__A, __B); 
}

__m128 test_mm_getexp_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_getexp_round_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_getexp_round_ss(__A, __B, 8); 
}

__m128 test_mm_getexp_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_getexp_ss
  // CHECK: @llvm.x86.avx512.mask.getexp.ss
  return _mm_getexp_ss(__A, __B); 
}

__m128d test_mm_getmant_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_getmant_round_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_getmant_round_sd(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src, 8); 
}

__m128d test_mm_getmant_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_getmant_sd
  // CHECK: @llvm.x86.avx512.mask.getmant.sd
  return _mm_getmant_sd(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src); 
}



__m128 test_mm_getmant_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_getmant_round_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_getmant_round_ss(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src, 8); 
}

__m128 test_mm_getmant_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_getmant_ss
  // CHECK: @llvm.x86.avx512.mask.getmant.ss
  return _mm_getmant_ss(__A, __B, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src); 
}

__mmask16 test_mm512_kmov(__mmask16 __A) {
  // CHECK-LABEL: @test_mm512_kmov
  // CHECK: load i16, i16* %__A.addr.i, align 2
  return _mm512_kmov(__A); 
}

__m512d test_mm512_mask_unpackhi_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_unpackhi_pd
  // CHECK: @llvm.x86.avx512.mask.unpckh.pd.512
  return _mm512_mask_unpackhi_pd(__W, __U, __A, __B); 
}
unsigned long long test_mm_cvt_roundsd_si64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvt_roundsd_si64
  // CHECK: @llvm.x86.avx512.vcvtsd2si64
  return _mm_cvt_roundsd_si64(__A, _MM_FROUND_CUR_DIRECTION); 
}
__m512i test_mm512_mask2_permutex2var_epi32(__m512i __A, __m512i __I, __mmask16 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.d.512
  return _mm512_mask2_permutex2var_epi32(__A, __I, __U, __B); 
}
__m512i test_mm512_unpackhi_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpackhi_epi32
  // CHECK: @llvm.x86.avx512.mask.punpckhd.q.512
  return _mm512_unpackhi_epi32(__A, __B); 
}

__m512d test_mm512_maskz_unpackhi_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpackhi_pd
  // CHECK: @llvm.x86.avx512.mask.unpckh.pd.512
  return _mm512_maskz_unpackhi_pd(__U, __A, __B); 
}
long long test_mm_cvt_roundsd_i64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvt_roundsd_i64
  // CHECK: @llvm.x86.avx512.vcvtsd2si64
  return _mm_cvt_roundsd_i64(__A, _MM_FROUND_CUR_DIRECTION); 
}
__m512d test_mm512_mask2_permutex2var_pd(__m512d __A, __m512i __I, __mmask8 __U, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_pd
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.pd.512
  return _mm512_mask2_permutex2var_pd(__A, __I, __U, __B); 
}
__m512i test_mm512_mask_unpackhi_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpackhi_epi32
  // CHECK: @llvm.x86.avx512.mask.punpckhd.q.512
  return _mm512_mask_unpackhi_epi32(__W, __U, __A, __B); 
}

__m512 test_mm512_mask_unpackhi_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_unpackhi_ps
  // CHECK: @llvm.x86.avx512.mask.unpckh.ps.512
  return _mm512_mask_unpackhi_ps(__W, __U, __A, __B); 
}

__m512 test_mm512_maskz_unpackhi_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpackhi_ps
  // CHECK: @llvm.x86.avx512.mask.unpckh.ps.512
  return _mm512_maskz_unpackhi_ps(__U, __A, __B); 
}

__m512d test_mm512_mask_unpacklo_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_unpacklo_pd
  // CHECK: @llvm.x86.avx512.mask.unpckl.pd.512
  return _mm512_mask_unpacklo_pd(__W, __U, __A, __B); 
}

__m512d test_mm512_maskz_unpacklo_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpacklo_pd
  // CHECK: @llvm.x86.avx512.mask.unpckl.pd.512
  return _mm512_maskz_unpacklo_pd(__U, __A, __B); 
}

__m512 test_mm512_mask_unpacklo_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_unpacklo_ps
  // CHECK: @llvm.x86.avx512.mask.unpckl.ps.512
  return _mm512_mask_unpacklo_ps(__W, __U, __A, __B); 
}

__m512 test_mm512_maskz_unpacklo_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpacklo_ps
  // CHECK: @llvm.x86.avx512.mask.unpckl.ps.512
  return _mm512_maskz_unpacklo_ps(__U, __A, __B); 
}
int test_mm_cvt_roundsd_si32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvt_roundsd_si32
  // CHECK: @llvm.x86.avx512.vcvtsd2si32
  return _mm_cvt_roundsd_si32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvt_roundsd_i32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvt_roundsd_i32
  // CHECK: @llvm.x86.avx512.vcvtsd2si32
  return _mm_cvt_roundsd_i32(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned test_mm_cvt_roundsd_u32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvt_roundsd_u32
  // CHECK: @llvm.x86.avx512.vcvtsd2usi32
  return _mm_cvt_roundsd_u32(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned test_mm_cvtsd_u32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtsd_u32
  // CHECK: @llvm.x86.avx512.vcvtsd2usi32
  return _mm_cvtsd_u32(__A); 
}

unsigned long long test_mm_cvt_roundsd_u64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvt_roundsd_u64
  // CHECK: @llvm.x86.avx512.vcvtsd2usi64
  return _mm_cvt_roundsd_u64(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned long long test_mm_cvtsd_u64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtsd_u64
  // CHECK: @llvm.x86.avx512.vcvtsd2usi64
  return _mm_cvtsd_u64(__A); 
}

int test_mm_cvt_roundss_si32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvt_roundss_si32
  // CHECK: @llvm.x86.avx512.vcvtss2si32
  return _mm_cvt_roundss_si32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvt_roundss_i32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvt_roundss_i32
  // CHECK: @llvm.x86.avx512.vcvtss2si32
  return _mm_cvt_roundss_i32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvt_roundss_si64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvt_roundss_si64
  // CHECK: @llvm.x86.avx512.vcvtss2si64
  return _mm_cvt_roundss_si64(__A, _MM_FROUND_CUR_DIRECTION); 
}

long long test_mm_cvt_roundss_i64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvt_roundss_i64
  // CHECK: @llvm.x86.avx512.vcvtss2si64
  return _mm_cvt_roundss_i64(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned test_mm_cvt_roundss_u32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvt_roundss_u32
  // CHECK: @llvm.x86.avx512.vcvtss2usi32
  return _mm_cvt_roundss_u32(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned test_mm_cvtss_u32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtss_u32
  // CHECK: @llvm.x86.avx512.vcvtss2usi32
  return _mm_cvtss_u32(__A); 
}

unsigned long long test_mm_cvt_roundss_u64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvt_roundss_u64
  // CHECK: @llvm.x86.avx512.vcvtss2usi64
  return _mm_cvt_roundss_u64(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned long long test_mm_cvtss_u64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtss_u64
  // CHECK: @llvm.x86.avx512.vcvtss2usi64
  return _mm_cvtss_u64(__A); 
}

int test_mm_cvtt_roundsd_i32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundsd_i32
  // CHECK: @llvm.x86.avx512.cvttsd2si
  return _mm_cvtt_roundsd_i32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvtt_roundsd_si32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundsd_si32
  // CHECK: @llvm.x86.avx512.cvttsd2si
  return _mm_cvtt_roundsd_si32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvttsd_i32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttsd_i32
  // CHECK: @llvm.x86.avx512.cvttsd2si
  return _mm_cvttsd_i32(__A); 
}

unsigned long long test_mm_cvtt_roundsd_si64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundsd_si64
  // CHECK: @llvm.x86.avx512.cvttsd2si64
  return _mm_cvtt_roundsd_si64(__A, _MM_FROUND_CUR_DIRECTION); 
}

long long test_mm_cvtt_roundsd_i64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundsd_i64
  // CHECK: @llvm.x86.avx512.cvttsd2si64
  return _mm_cvtt_roundsd_i64(__A, _MM_FROUND_CUR_DIRECTION); 
}

long long test_mm_cvttsd_i64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttsd_i64
  // CHECK: @llvm.x86.avx512.cvttsd2si64
  return _mm_cvttsd_i64(__A); 
}

unsigned test_mm_cvtt_roundsd_u32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundsd_u32
  // CHECK: @llvm.x86.avx512.cvttsd2usi
  return _mm_cvtt_roundsd_u32(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned test_mm_cvttsd_u32(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttsd_u32
  // CHECK: @llvm.x86.avx512.cvttsd2usi
  return _mm_cvttsd_u32(__A); 
}

unsigned long long test_mm_cvtt_roundsd_u64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundsd_u64
  // CHECK: @llvm.x86.avx512.cvttsd2usi64
  return _mm_cvtt_roundsd_u64(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned long long test_mm_cvttsd_u64(__m128d __A) {
  // CHECK-LABEL: @test_mm_cvttsd_u64
  // CHECK: @llvm.x86.avx512.cvttsd2usi64
  return _mm_cvttsd_u64(__A); 
}

int test_mm_cvtt_roundss_i32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundss_i32
  // CHECK: @llvm.x86.avx512.cvttss2si
  return _mm_cvtt_roundss_i32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvtt_roundss_si32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundss_si32
  // CHECK: @llvm.x86.avx512.cvttss2si
  return _mm_cvtt_roundss_si32(__A, _MM_FROUND_CUR_DIRECTION); 
}

int test_mm_cvttss_i32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttss_i32
  // CHECK: @llvm.x86.avx512.cvttss2si
  return _mm_cvttss_i32(__A); 
}

float test_mm_cvtt_roundss_i64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundss_i64
  // CHECK: @llvm.x86.avx512.cvttss2si64
  return _mm_cvtt_roundss_i64(__A, _MM_FROUND_CUR_DIRECTION); 
}

long long test_mm_cvtt_roundss_si64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundss_si64
  // CHECK: @llvm.x86.avx512.cvttss2si64
  return _mm_cvtt_roundss_si64(__A, _MM_FROUND_CUR_DIRECTION); 
}

long long test_mm_cvttss_i64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttss_i64
  // CHECK: @llvm.x86.avx512.cvttss2si64
  return _mm_cvttss_i64(__A); 
}

unsigned test_mm_cvtt_roundss_u32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundss_u32
  // CHECK: @llvm.x86.avx512.cvttss2usi
  return _mm_cvtt_roundss_u32(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned test_mm_cvttss_u32(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttss_u32
  // CHECK: @llvm.x86.avx512.cvttss2usi
  return _mm_cvttss_u32(__A); 
}

unsigned long long test_mm_cvtt_roundss_u64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvtt_roundss_u64
  // CHECK: @llvm.x86.avx512.cvttss2usi64
  return _mm_cvtt_roundss_u64(__A, _MM_FROUND_CUR_DIRECTION); 
}

unsigned long long test_mm_cvttss_u64(__m128 __A) {
  // CHECK-LABEL: @test_mm_cvttss_u64
  // CHECK: @llvm.x86.avx512.cvttss2usi64
  return _mm_cvttss_u64(__A); 
}
__m512 test_mm512_mask2_permutex2var_ps(__m512 __A, __m512i __I, __mmask16 __U, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_ps
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.ps.512
  return _mm512_mask2_permutex2var_ps(__A, __I, __U, __B); 
}

__m512i test_mm512_mask2_permutex2var_epi64(__m512i __A, __m512i __I, __mmask8 __U, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask2_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.mask.vpermi2var.q.512
  return _mm512_mask2_permutex2var_epi64(__A, __I, __U, __B); 
}

__m512d test_mm512_permute_pd(__m512d __X) {
  // CHECK-LABEL: @test_mm512_permute_pd
  // CHECK: @llvm.x86.avx512.mask.vpermil.pd.512
  return _mm512_permute_pd(__X, 2); 
}

__m512d test_mm512_mask_permute_pd(__m512d __W, __mmask8 __U, __m512d __X) {
  // CHECK-LABEL: @test_mm512_mask_permute_pd
  // CHECK: @llvm.x86.avx512.mask.vpermil.pd.512
  return _mm512_mask_permute_pd(__W, __U, __X, 2); 
}

__m512d test_mm512_maskz_permute_pd(__mmask8 __U, __m512d __X) {
  // CHECK-LABEL: @test_mm512_maskz_permute_pd
  // CHECK: @llvm.x86.avx512.mask.vpermil.pd.512
  return _mm512_maskz_permute_pd(__U, __X, 2); 
}

__m512 test_mm512_permute_ps(__m512 __X) {
  // CHECK-LABEL: @test_mm512_permute_ps
  // CHECK: @llvm.x86.avx512.mask.vpermil.ps.512
  return _mm512_permute_ps(__X, 2); 
}

__m512 test_mm512_mask_permute_ps(__m512 __W, __mmask16 __U, __m512 __X) {
  // CHECK-LABEL: @test_mm512_mask_permute_ps
  // CHECK: @llvm.x86.avx512.mask.vpermil.ps.512
  return _mm512_mask_permute_ps(__W, __U, __X, 2); 
}

__m512 test_mm512_maskz_permute_ps(__mmask16 __U, __m512 __X) {
  // CHECK-LABEL: @test_mm512_maskz_permute_ps
  // CHECK: @llvm.x86.avx512.mask.vpermil.ps.512
  return _mm512_maskz_permute_ps(__U, __X, 2); 
}

__m512d test_mm512_permutevar_pd(__m512d __A, __m512i __C) {
  // CHECK-LABEL: @test_mm512_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd.512
  return _mm512_permutevar_pd(__A, __C); 
}

__m512d test_mm512_mask_permutevar_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd.512
  return _mm512_mask_permutevar_pd(__W, __U, __A, __C); 
}

__m512d test_mm512_maskz_permutevar_pd(__mmask8 __U, __m512d __A, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_permutevar_pd
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.pd.512
  return _mm512_maskz_permutevar_pd(__U, __A, __C); 
}

__m512 test_mm512_permutevar_ps(__m512 __A, __m512i __C) {
  // CHECK-LABEL: @test_mm512_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps.512
  return _mm512_permutevar_ps(__A, __C); 
}

__m512 test_mm512_mask_permutevar_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps.512
  return _mm512_mask_permutevar_ps(__W, __U, __A, __C); 
}

__m512 test_mm512_maskz_permutevar_ps(__mmask16 __U, __m512 __A, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_permutevar_ps
  // CHECK: @llvm.x86.avx512.mask.vpermilvar.ps.512
  return _mm512_maskz_permutevar_ps(__U, __A, __C); 
}

__m512i test_mm512_maskz_permutex2var_epi32(__mmask16 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_epi32
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.d.512
  return _mm512_maskz_permutex2var_epi32(__U, __A, __I, __B); 
}

__m512d test_mm512_maskz_permutex2var_pd(__mmask8 __U, __m512d __A, __m512i __I, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_pd
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.pd.512
  return _mm512_maskz_permutex2var_pd(__U, __A, __I, __B); 
}

__m512 test_mm512_maskz_permutex2var_ps(__mmask16 __U, __m512 __A, __m512i __I, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_ps
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.ps.512
  return _mm512_maskz_permutex2var_ps(__U, __A, __I, __B); 
}

__m512i test_mm512_maskz_permutex2var_epi64(__mmask8 __U, __m512i __A, __m512i __I, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_permutex2var_epi64
  // CHECK: @llvm.x86.avx512.maskz.vpermt2var.q.512
  return _mm512_maskz_permutex2var_epi64(__U, __A, __I, __B);
}
__mmask16 test_mm512_testn_epi32_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_testn_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestnm.d.512
  return _mm512_testn_epi32_mask(__A, __B); 
}

__mmask16 test_mm512_mask_testn_epi32_mask(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_testn_epi32_mask
  // CHECK: @llvm.x86.avx512.ptestnm.d.512
  return _mm512_mask_testn_epi32_mask(__U, __A, __B); 
}

__mmask8 test_mm512_testn_epi64_mask(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_testn_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestnm.q.512
  return _mm512_testn_epi64_mask(__A, __B); 
}

__mmask8 test_mm512_mask_testn_epi64_mask(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_testn_epi64_mask
  // CHECK: @llvm.x86.avx512.ptestnm.q.512
  return _mm512_mask_testn_epi64_mask(__U, __A, __B); 
}
__m512i test_mm512_maskz_unpackhi_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpackhi_epi32
  // CHECK: @llvm.x86.avx512.mask.punpckhd.q.512
  return _mm512_maskz_unpackhi_epi32(__U, __A, __B); 
}

__m512i test_mm512_unpackhi_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpackhi_epi64
  // CHECK: @llvm.x86.avx512.mask.punpckhqd.q.512
  return _mm512_unpackhi_epi64(__A, __B); 
}

__m512i test_mm512_mask_unpackhi_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpackhi_epi64
  // CHECK: @llvm.x86.avx512.mask.punpckhqd.q.512
  return _mm512_mask_unpackhi_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpackhi_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpackhi_epi64
  // CHECK: @llvm.x86.avx512.mask.punpckhqd.q.512
  return _mm512_maskz_unpackhi_epi64(__U, __A, __B); 
}

__m512i test_mm512_unpacklo_epi32(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpacklo_epi32
  // CHECK: @llvm.x86.avx512.mask.punpckld.q.512
  return _mm512_unpacklo_epi32(__A, __B); 
}

__m512i test_mm512_mask_unpacklo_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpacklo_epi32
  // CHECK: @llvm.x86.avx512.mask.punpckld.q.512
  return _mm512_mask_unpacklo_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpacklo_epi32(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpacklo_epi32
  // CHECK: @llvm.x86.avx512.mask.punpckld.q.512
  return _mm512_maskz_unpacklo_epi32(__U, __A, __B); 
}

__m512i test_mm512_unpacklo_epi64(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_unpacklo_epi64
  // CHECK: @llvm.x86.avx512.mask.punpcklqd.q.512
  return _mm512_unpacklo_epi64(__A, __B); 
}

__m512i test_mm512_mask_unpacklo_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_unpacklo_epi64
  // CHECK: @llvm.x86.avx512.mask.punpcklqd.q.512
  return _mm512_mask_unpacklo_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_unpacklo_epi64(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_unpacklo_epi64
  // CHECK: @llvm.x86.avx512.mask.punpcklqd.q.512
  return _mm512_maskz_unpacklo_epi64(__U, __A, __B); 
}

__m128d test_mm_roundscale_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_roundscale_round_sd
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
  return _mm_roundscale_round_sd(__A, __B, 3, _MM_FROUND_CUR_DIRECTION); 
}

__m128d test_mm_roundscale_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_roundscale_sd
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
  return _mm_roundscale_sd(__A, __B, 3); 
}

__m128d test_mm_mask_roundscale_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_mask_roundscale_sd(__W,__U,__A,__B,3);
}

__m128d test_mm_mask_roundscale_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_mask_roundscale_round_sd(__W,__U,__A,__B,3,_MM_FROUND_CUR_DIRECTION);
}

__m128d test_mm_maskz_roundscale_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_maskz_roundscale_sd(__U,__A,__B,3);
}

__m128d test_mm_maskz_roundscale_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.rndscale.sd
    return _mm_maskz_roundscale_round_sd(__U,__A,__B,3,_MM_FROUND_CUR_DIRECTION );
}

__m128 test_mm_roundscale_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_roundscale_round_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
  return _mm_roundscale_round_ss(__A, __B, 3, _MM_FROUND_CUR_DIRECTION); 
}

__m128 test_mm_roundscale_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_roundscale_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
  return _mm_roundscale_ss(__A, __B, 3); 
}

__m128d test_mm_mask_roundscale_ss(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_mask_roundscale_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
    return _mm_mask_roundscale_ss(__W,__U,__A,__B,3);
}

__m128d test_mm_maskz_roundscale_round_ss( __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_maskz_roundscale_round_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
    return _mm_maskz_roundscale_round_ss(__U,__A,__B,3,_MM_FROUND_CUR_DIRECTION);
}

__m128d test_mm_maskz_roundscale_ss(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_maskz_roundscale_ss
  // CHECK: @llvm.x86.avx512.mask.rndscale.ss
    return _mm_maskz_roundscale_ss(__U,__A,__B,3);
}

__m512d test_mm512_scalef_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_scalef_round_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_scalef_round_pd(__A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_mask_scalef_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_scalef_round_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_mask_scalef_round_pd(__W, __U, __A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_maskz_scalef_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_scalef_round_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_maskz_scalef_round_pd(__U, __A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_scalef_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_scalef_pd(__A, __B); 
}

__m512d test_mm512_mask_scalef_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_mask_scalef_pd(__W, __U, __A, __B); 
}

__m512d test_mm512_maskz_scalef_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_scalef_pd
  // CHECK: @llvm.x86.avx512.mask.scalef.pd.512
  return _mm512_maskz_scalef_pd(__U, __A, __B); 
}

__m512 test_mm512_scalef_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_scalef_round_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_scalef_round_ps(__A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_mask_scalef_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_scalef_round_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_mask_scalef_round_ps(__W, __U, __A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_maskz_scalef_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_scalef_round_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_maskz_scalef_round_ps(__U, __A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_scalef_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_scalef_ps(__A, __B); 
}

__m512 test_mm512_mask_scalef_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_mask_scalef_ps(__W, __U, __A, __B); 
}

__m512 test_mm512_maskz_scalef_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_scalef_ps
  // CHECK: @llvm.x86.avx512.mask.scalef.ps.512
  return _mm512_maskz_scalef_ps(__U, __A, __B); 
}

__m128d test_mm_scalef_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_scalef_round_sd
  // CHECK: @llvm.x86.avx512.mask.scalef
  return _mm_scalef_round_sd(__A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m128d test_mm_scalef_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_scalef_sd
  // CHECK: @llvm.x86.avx512.mask.scalef
  return _mm_scalef_sd(__A, __B); 
}

__m128d test_mm_mask_scalef_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_mask_scalef_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd
  return _mm_mask_scalef_sd(__W, __U, __A, __B);
}

__m128d test_mm_mask_scalef_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_mask_scalef_round_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd
    return _mm_mask_scalef_round_sd(__W, __U, __A, __B, _MM_FROUND_CUR_DIRECTION);
}

__m128d test_mm_maskz_scalef_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_maskz_scalef_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd
    return _mm_maskz_scalef_sd(__U, __A, __B);
}

__m128d test_mm_maskz_scalef_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK-LABEL: @test_mm_maskz_scalef_round_sd
  // CHECK: @llvm.x86.avx512.mask.scalef.sd
    return _mm_maskz_scalef_round_sd(__U, __A, __B, _MM_FROUND_CUR_DIRECTION);
}

__m128 test_mm_scalef_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_scalef_round_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
  return _mm_scalef_round_ss(__A, __B, _MM_FROUND_CUR_DIRECTION); 
}

__m128 test_mm_scalef_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_scalef_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
  return _mm_scalef_ss(__A, __B); 
}

__m128 test_mm_mask_scalef_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: @test_mm_mask_scalef_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
    return _mm_mask_scalef_ss(__W, __U, __A, __B);
}

__m128 test_mm_mask_scalef_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: @test_mm_mask_scalef_round_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
    return _mm_mask_scalef_round_ss(__W, __U, __A, __B, _MM_FROUND_CUR_DIRECTION);
}

__m128 test_mm_maskz_scalef_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: @test_mm_maskz_scalef_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
    return _mm_maskz_scalef_ss(__U, __A, __B);
}

__m128 test_mm_maskz_scalef_round_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK-LABEL: @test_mm_maskz_scalef_round_ss
  // CHECK: @llvm.x86.avx512.mask.scalef.ss
    return _mm_maskz_scalef_round_ss(__U, __A, __B, _MM_FROUND_CUR_DIRECTION);
}

__m512i test_mm512_srai_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.512
  return _mm512_srai_epi32(__A, 5); 
}

__m512i test_mm512_mask_srai_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.512
  return _mm512_mask_srai_epi32(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_srai_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_srai_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.di.512
  return _mm512_maskz_srai_epi32(__U, __A, 5); 
}

__m512i test_mm512_srai_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.512
  return _mm512_srai_epi64(__A, 5); 
}

__m512i test_mm512_mask_srai_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.512
  return _mm512_mask_srai_epi64(__W, __U, __A, 5); 
}

__m512i test_mm512_maskz_srai_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_srai_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.qi.512
  return _mm512_maskz_srai_epi64(__U, __A, 5); 
}

__m512i test_mm512_sll_epi32(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_sll_epi32
  // CHECK: @llvm.x86.avx512.mask.psll.d
  return _mm512_sll_epi32(__A, __B); 
}

__m512i test_mm512_mask_sll_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_sll_epi32
  // CHECK: @llvm.x86.avx512.mask.psll.d
  return _mm512_mask_sll_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sll_epi32(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sll_epi32
  // CHECK: @llvm.x86.avx512.mask.psll.d
  return _mm512_maskz_sll_epi32(__U, __A, __B); 
}

__m512i test_mm512_sll_epi64(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_sll_epi64
  // CHECK: @llvm.x86.avx512.mask.psll.q
  return _mm512_sll_epi64(__A, __B); 
}

__m512i test_mm512_mask_sll_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_sll_epi64
  // CHECK: @llvm.x86.avx512.mask.psll.q
  return _mm512_mask_sll_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sll_epi64(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sll_epi64
  // CHECK: @llvm.x86.avx512.mask.psll.q
  return _mm512_maskz_sll_epi64(__U, __A, __B); 
}

__m512i test_mm512_sllv_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv.d
  return _mm512_sllv_epi32(__X, __Y); 
}

__m512i test_mm512_mask_sllv_epi32(__m512i __W, __mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv.d
  return _mm512_mask_sllv_epi32(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_sllv_epi32(__mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_sllv_epi32
  // CHECK: @llvm.x86.avx512.mask.psllv.d
  return _mm512_maskz_sllv_epi32(__U, __X, __Y); 
}

__m512i test_mm512_sllv_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv.q
  return _mm512_sllv_epi64(__X, __Y); 
}

__m512i test_mm512_mask_sllv_epi64(__m512i __W, __mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv.q
  return _mm512_mask_sllv_epi64(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_sllv_epi64(__mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_sllv_epi64
  // CHECK: @llvm.x86.avx512.mask.psllv.q
  return _mm512_maskz_sllv_epi64(__U, __X, __Y); 
}

__m512i test_mm512_sra_epi32(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d
  return _mm512_sra_epi32(__A, __B); 
}

__m512i test_mm512_mask_sra_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d
  return _mm512_mask_sra_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sra_epi32(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sra_epi32
  // CHECK: @llvm.x86.avx512.mask.psra.d
  return _mm512_maskz_sra_epi32(__U, __A, __B); 
}

__m512i test_mm512_sra_epi64(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q
  return _mm512_sra_epi64(__A, __B); 
}

__m512i test_mm512_mask_sra_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q
  return _mm512_mask_sra_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_sra_epi64(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_sra_epi64
  // CHECK: @llvm.x86.avx512.mask.psra.q
  return _mm512_maskz_sra_epi64(__U, __A, __B); 
}

__m512i test_mm512_srav_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav.d
  return _mm512_srav_epi32(__X, __Y); 
}

__m512i test_mm512_mask_srav_epi32(__m512i __W, __mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav.d
  return _mm512_mask_srav_epi32(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srav_epi32(__mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_srav_epi32
  // CHECK: @llvm.x86.avx512.mask.psrav.d
  return _mm512_maskz_srav_epi32(__U, __X, __Y); 
}

__m512i test_mm512_srav_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q
  return _mm512_srav_epi64(__X, __Y); 
}

__m512i test_mm512_mask_srav_epi64(__m512i __W, __mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q
  return _mm512_mask_srav_epi64(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srav_epi64(__mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_srav_epi64
  // CHECK: @llvm.x86.avx512.mask.psrav.q
  return _mm512_maskz_srav_epi64(__U, __X, __Y); 
}

__m512i test_mm512_srl_epi32(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d
  return _mm512_srl_epi32(__A, __B); 
}

__m512i test_mm512_mask_srl_epi32(__m512i __W, __mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d
  return _mm512_mask_srl_epi32(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srl_epi32(__mmask16 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_srl_epi32
  // CHECK: @llvm.x86.avx512.mask.psrl.d
  return _mm512_maskz_srl_epi32(__U, __A, __B); 
}

__m512i test_mm512_srl_epi64(__m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q
  return _mm512_srl_epi64(__A, __B); 
}

__m512i test_mm512_mask_srl_epi64(__m512i __W, __mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_mask_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q
  return _mm512_mask_srl_epi64(__W, __U, __A, __B); 
}

__m512i test_mm512_maskz_srl_epi64(__mmask8 __U, __m512i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm512_maskz_srl_epi64
  // CHECK: @llvm.x86.avx512.mask.psrl.q
  return _mm512_maskz_srl_epi64(__U, __A, __B); 
}

__m512i test_mm512_srlv_epi32(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv.d
  return _mm512_srlv_epi32(__X, __Y); 
}

__m512i test_mm512_mask_srlv_epi32(__m512i __W, __mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv.d
  return _mm512_mask_srlv_epi32(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srlv_epi32(__mmask16 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_srlv_epi32
  // CHECK: @llvm.x86.avx512.mask.psrlv.d
  return _mm512_maskz_srlv_epi32(__U, __X, __Y); 
}

__m512i test_mm512_srlv_epi64(__m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv.q
  return _mm512_srlv_epi64(__X, __Y); 
}

__m512i test_mm512_mask_srlv_epi64(__m512i __W, __mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_mask_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv.q
  return _mm512_mask_srlv_epi64(__W, __U, __X, __Y); 
}

__m512i test_mm512_maskz_srlv_epi64(__mmask8 __U, __m512i __X, __m512i __Y) {
  // CHECK-LABEL: @test_mm512_maskz_srlv_epi64
  // CHECK: @llvm.x86.avx512.mask.psrlv.q
  return _mm512_maskz_srlv_epi64(__U, __X, __Y); 
}

__m512i test_mm512_ternarylogic_epi32(__m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.mask.pternlog.d.512
  return _mm512_ternarylogic_epi32(__A, __B, __C, 4); 
}

__m512i test_mm512_mask_ternarylogic_epi32(__m512i __A, __mmask16 __U, __m512i __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.mask.pternlog.d.512
  return _mm512_mask_ternarylogic_epi32(__A, __U, __B, __C, 4); 
}

__m512i test_mm512_maskz_ternarylogic_epi32(__mmask16 __U, __m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_ternarylogic_epi32
  // CHECK: @llvm.x86.avx512.maskz.pternlog.d.512
  return _mm512_maskz_ternarylogic_epi32(__U, __A, __B, __C, 4); 
}

__m512i test_mm512_ternarylogic_epi64(__m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.mask.pternlog.q.512
  return _mm512_ternarylogic_epi64(__A, __B, __C, 4); 
}

__m512i test_mm512_mask_ternarylogic_epi64(__m512i __A, __mmask8 __U, __m512i __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_mask_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.mask.pternlog.q.512
  return _mm512_mask_ternarylogic_epi64(__A, __U, __B, __C, 4); 
}

__m512i test_mm512_maskz_ternarylogic_epi64(__mmask8 __U, __m512i __A, __m512i __B, __m512i __C) {
  // CHECK-LABEL: @test_mm512_maskz_ternarylogic_epi64
  // CHECK: @llvm.x86.avx512.maskz.pternlog.q.512
  return _mm512_maskz_ternarylogic_epi64(__U, __A, __B, __C, 4); 
}

__m512 test_mm512_shuffle_f32x4(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_shuffle_f32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.f32x4
  return _mm512_shuffle_f32x4(__A, __B, 4); 
}

__m512 test_mm512_mask_shuffle_f32x4(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_f32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.f32x4
  return _mm512_mask_shuffle_f32x4(__W, __U, __A, __B, 4); 
}

__m512 test_mm512_maskz_shuffle_f32x4(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_f32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.f32x4
  return _mm512_maskz_shuffle_f32x4(__U, __A, __B, 4); 
}

__m512d test_mm512_shuffle_f64x2(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_shuffle_f64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.f64x2
  return _mm512_shuffle_f64x2(__A, __B, 4); 
}

__m512d test_mm512_mask_shuffle_f64x2(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_f64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.f64x2
  return _mm512_mask_shuffle_f64x2(__W, __U, __A, __B, 4); 
}

__m512d test_mm512_maskz_shuffle_f64x2(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_f64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.f64x2
  return _mm512_maskz_shuffle_f64x2(__U, __A, __B, 4); 
}

__m512i test_mm512_shuffle_i32x4(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shuffle_i32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.i32x4
  return _mm512_shuffle_i32x4(__A, __B, 4); 
}

__m512i test_mm512_mask_shuffle_i32x4(__m512i __W, __mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_i32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.i32x4
  return _mm512_mask_shuffle_i32x4(__W, __U, __A, __B, 4); 
}

__m512i test_mm512_maskz_shuffle_i32x4(__mmask16 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_i32x4
  // CHECK: @llvm.x86.avx512.mask.shuf.i32x4
  return _mm512_maskz_shuffle_i32x4(__U, __A, __B, 4); 
}

__m512i test_mm512_shuffle_i64x2(__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_shuffle_i64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.i64x2
  return _mm512_shuffle_i64x2(__A, __B, 4); 
}

__m512i test_mm512_mask_shuffle_i64x2(__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_i64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.i64x2
  return _mm512_mask_shuffle_i64x2(__W, __U, __A, __B, 4); 
}

__m512i test_mm512_maskz_shuffle_i64x2(__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_i64x2
  // CHECK: @llvm.x86.avx512.mask.shuf.i64x2
  return _mm512_maskz_shuffle_i64x2(__U, __A, __B, 4); 
}

__m512d test_mm512_shuffle_pd(__m512d __M, __m512d __V) {
  // CHECK-LABEL: @test_mm512_shuffle_pd
  // CHECK: @llvm.x86.avx512.mask.shuf.pd.512
  return _mm512_shuffle_pd(__M, __V, 4); 
}

__m512d test_mm512_mask_shuffle_pd(__m512d __W, __mmask8 __U, __m512d __M, __m512d __V) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_pd
  // CHECK: @llvm.x86.avx512.mask.shuf.pd.512
  return _mm512_mask_shuffle_pd(__W, __U, __M, __V, 4); 
}

__m512d test_mm512_maskz_shuffle_pd(__mmask8 __U, __m512d __M, __m512d __V) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_pd
  // CHECK: @llvm.x86.avx512.mask.shuf.pd.512
  return _mm512_maskz_shuffle_pd(__U, __M, __V, 4); 
}

__m512 test_mm512_shuffle_ps(__m512 __M, __m512 __V) {
  // CHECK-LABEL: @test_mm512_shuffle_ps
  // CHECK: @llvm.x86.avx512.mask.shuf.ps.512
  return _mm512_shuffle_ps(__M, __V, 4); 
}

__m512 test_mm512_mask_shuffle_ps(__m512 __W, __mmask16 __U, __m512 __M, __m512 __V) {
  // CHECK-LABEL: @test_mm512_mask_shuffle_ps
  // CHECK: @llvm.x86.avx512.mask.shuf.ps.512
  return _mm512_mask_shuffle_ps(__W, __U, __M, __V, 4); 
}

__m512 test_mm512_maskz_shuffle_ps(__mmask16 __U, __m512 __M, __m512 __V) {
  // CHECK-LABEL: @test_mm512_maskz_shuffle_ps
  // CHECK: @llvm.x86.avx512.mask.shuf.ps.512
  return _mm512_maskz_shuffle_ps(__U, __M, __V, 4); 
}

__m128d test_mm_sqrt_round_sd(__m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_sqrt_round_sd
  // CHECK: @llvm.x86.avx512.mask.sqrt.sd
  return _mm_sqrt_round_sd(__A, __B, 4); 
}

__m128d test_mm_mask_sqrt_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.sd
    return _mm_mask_sqrt_sd(__W,__U,__A,__B);
}

__m128d test_mm_mask_sqrt_round_sd(__m128d __W, __mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.sd
    return _mm_mask_sqrt_round_sd(__W,__U,__A,__B,_MM_FROUND_CUR_DIRECTION);
}

__m128d test_mm_maskz_sqrt_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.sd
    return _mm_maskz_sqrt_sd(__U,__A,__B);
}

__m128d test_mm_maskz_sqrt_round_sd(__mmask8 __U, __m128d __A, __m128d __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.sd
    return _mm_maskz_sqrt_round_sd(__U,__A,__B,_MM_FROUND_CUR_DIRECTION);
}

__m128 test_mm_sqrt_round_ss(__m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_sqrt_round_ss
  // CHECK: @llvm.x86.avx512.mask.sqrt.ss
  return _mm_sqrt_round_ss(__A, __B, 4); 
}

__m128 test_mm_mask_sqrt_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.ss
    return _mm_mask_sqrt_ss(__W,__U,__A,__B);
}

__m128 test_mm_mask_sqrt_round_ss(__m128 __W, __mmask8 __U, __m128 __A, __m128 __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.ss
    return _mm_mask_sqrt_round_ss(__W,__U,__A,__B,_MM_FROUND_CUR_DIRECTION);
}

__m128 test_mm_maskz_sqrt_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.ss
    return _mm_maskz_sqrt_ss(__U,__A,__B);
}

__m128 test_mm_maskz_sqrt_round_ss(__mmask8 __U, __m128 __A, __m128 __B){
  // CHECK: @llvm.x86.avx512.mask.sqrt.ss
    return _mm_maskz_sqrt_round_ss(__U,__A,__B,_MM_FROUND_CUR_DIRECTION);
}

__m128i test_mm512_cvtsepi32_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.512
  return _mm512_cvtsepi32_epi8(__A); 
}

__m128i test_mm512_mask_cvtsepi32_epi8(__m128i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.512
  return _mm512_mask_cvtsepi32_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtsepi32_epi8(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtsepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.512
  return _mm512_maskz_cvtsepi32_epi8(__M, __A); 
}

void test_mm512_mask_cvtsepi32_storeu_epi8(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.db.mem.512
  return _mm512_mask_cvtsepi32_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtsepi32_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.512
  return _mm512_cvtsepi32_epi16(__A); 
}

__m256i test_mm512_mask_cvtsepi32_epi16(__m256i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.512
  return _mm512_mask_cvtsepi32_epi16(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtsepi32_epi16(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtsepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.512
  return _mm512_maskz_cvtsepi32_epi16(__M, __A); 
}

void test_mm512_mask_cvtsepi32_storeu_epi16(void *__P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.dw.mem.512
  return _mm512_mask_cvtsepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtsepi64_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.512
  return _mm512_cvtsepi64_epi8(__A); 
}

__m128i test_mm512_mask_cvtsepi64_epi8(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.512
  return _mm512_mask_cvtsepi64_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtsepi64_epi8(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtsepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.512
  return _mm512_maskz_cvtsepi64_epi8(__M, __A); 
}

void test_mm512_mask_cvtsepi64_storeu_epi8(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovs.qb.mem.512
  return _mm512_mask_cvtsepi64_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtsepi64_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.512
  return _mm512_cvtsepi64_epi32(__A); 
}

__m256i test_mm512_mask_cvtsepi64_epi32(__m256i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.512
  return _mm512_mask_cvtsepi64_epi32(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtsepi64_epi32(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtsepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.512
  return _mm512_maskz_cvtsepi64_epi32(__M, __A); 
}

void test_mm512_mask_cvtsepi64_storeu_epi32(void *__P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovs.qd.mem.512
  return _mm512_mask_cvtsepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm512_cvtsepi64_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.512
  return _mm512_cvtsepi64_epi16(__A); 
}

__m128i test_mm512_mask_cvtsepi64_epi16(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.512
  return _mm512_mask_cvtsepi64_epi16(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtsepi64_epi16(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtsepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.512
  return _mm512_maskz_cvtsepi64_epi16(__M, __A); 
}

void test_mm512_mask_cvtsepi64_storeu_epi16(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtsepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovs.qw.mem.512
  return _mm512_mask_cvtsepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtusepi32_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.512
  return _mm512_cvtusepi32_epi8(__A); 
}

__m128i test_mm512_mask_cvtusepi32_epi8(__m128i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.512
  return _mm512_mask_cvtusepi32_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtusepi32_epi8(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtusepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.512
  return _mm512_maskz_cvtusepi32_epi8(__M, __A); 
}

void test_mm512_mask_cvtusepi32_storeu_epi8(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.db.mem.512
  return _mm512_mask_cvtusepi32_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtusepi32_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.512
  return _mm512_cvtusepi32_epi16(__A); 
}

__m256i test_mm512_mask_cvtusepi32_epi16(__m256i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.512
  return _mm512_mask_cvtusepi32_epi16(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtusepi32_epi16(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtusepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.512
  return _mm512_maskz_cvtusepi32_epi16(__M, __A); 
}

void test_mm512_mask_cvtusepi32_storeu_epi16(void *__P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.dw.mem.512
  return _mm512_mask_cvtusepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtusepi64_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.512
  return _mm512_cvtusepi64_epi8(__A); 
}

__m128i test_mm512_mask_cvtusepi64_epi8(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.512
  return _mm512_mask_cvtusepi64_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtusepi64_epi8(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtusepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.512
  return _mm512_maskz_cvtusepi64_epi8(__M, __A); 
}

void test_mm512_mask_cvtusepi64_storeu_epi8(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmovus.qb.mem.512
  return _mm512_mask_cvtusepi64_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtusepi64_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.512
  return _mm512_cvtusepi64_epi32(__A); 
}

__m256i test_mm512_mask_cvtusepi64_epi32(__m256i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.512
  return _mm512_mask_cvtusepi64_epi32(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtusepi64_epi32(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtusepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.512
  return _mm512_maskz_cvtusepi64_epi32(__M, __A); 
}

void test_mm512_mask_cvtusepi64_storeu_epi32(void* __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmovus.qd.mem.512
  return _mm512_mask_cvtusepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm512_cvtusepi64_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.512
  return _mm512_cvtusepi64_epi16(__A); 
}

__m128i test_mm512_mask_cvtusepi64_epi16(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.512
  return _mm512_mask_cvtusepi64_epi16(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtusepi64_epi16(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtusepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.512
  return _mm512_maskz_cvtusepi64_epi16(__M, __A); 
}

void test_mm512_mask_cvtusepi64_storeu_epi16(void *__P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtusepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmovus.qw.mem.512
  return _mm512_mask_cvtusepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtepi32_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.512
  return _mm512_cvtepi32_epi8(__A); 
}

__m128i test_mm512_mask_cvtepi32_epi8(__m128i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.512
  return _mm512_mask_cvtepi32_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtepi32_epi8(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi32_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.512
  return _mm512_maskz_cvtepi32_epi8(__M, __A); 
}

void test_mm512_mask_cvtepi32_storeu_epi8(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi32_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.db.mem.512
  return _mm512_mask_cvtepi32_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtepi32_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.512
  return _mm512_cvtepi32_epi16(__A); 
}

__m256i test_mm512_mask_cvtepi32_epi16(__m256i __O, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.512
  return _mm512_mask_cvtepi32_epi16(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtepi32_epi16(__mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi32_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.512
  return _mm512_maskz_cvtepi32_epi16(__M, __A); 
}

void test_mm512_mask_cvtepi32_storeu_epi16(void * __P, __mmask16 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi32_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.dw.mem.512
  return _mm512_mask_cvtepi32_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_cvtepi64_epi8(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.512
  return _mm512_cvtepi64_epi8(__A); 
}

__m128i test_mm512_mask_cvtepi64_epi8(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.512
  return _mm512_mask_cvtepi64_epi8(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtepi64_epi8(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi64_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.512
  return _mm512_maskz_cvtepi64_epi8(__M, __A); 
}

void test_mm512_mask_cvtepi64_storeu_epi8(void * __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_storeu_epi8
  // CHECK: @llvm.x86.avx512.mask.pmov.qb.mem.512
  return _mm512_mask_cvtepi64_storeu_epi8(__P, __M, __A); 
}

__m256i test_mm512_cvtepi64_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.512
  return _mm512_cvtepi64_epi32(__A); 
}

__m256i test_mm512_mask_cvtepi64_epi32(__m256i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.512
  return _mm512_mask_cvtepi64_epi32(__O, __M, __A); 
}

__m256i test_mm512_maskz_cvtepi64_epi32(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi64_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.512
  return _mm512_maskz_cvtepi64_epi32(__M, __A); 
}

void test_mm512_mask_cvtepi64_storeu_epi32(void* __P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_storeu_epi32
  // CHECK: @llvm.x86.avx512.mask.pmov.qd.mem.512
  return _mm512_mask_cvtepi64_storeu_epi32(__P, __M, __A); 
}

__m128i test_mm512_cvtepi64_epi16(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.512
  return _mm512_cvtepi64_epi16(__A); 
}

__m128i test_mm512_mask_cvtepi64_epi16(__m128i __O, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.512
  return _mm512_mask_cvtepi64_epi16(__O, __M, __A); 
}

__m128i test_mm512_maskz_cvtepi64_epi16(__mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi64_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.512
  return _mm512_maskz_cvtepi64_epi16(__M, __A); 
}

void test_mm512_mask_cvtepi64_storeu_epi16(void *__P, __mmask8 __M, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_storeu_epi16
  // CHECK: @llvm.x86.avx512.mask.pmov.qw.mem.512
  return _mm512_mask_cvtepi64_storeu_epi16(__P, __M, __A); 
}

__m128i test_mm512_extracti32x4_epi32(__m512i __A) {
  // CHECK-LABEL: @test_mm512_extracti32x4_epi32
  // CHECK: @llvm.x86.avx512.mask.vextracti32x4
  return _mm512_extracti32x4_epi32(__A, 3); 
}

__m128i test_mm512_mask_extracti32x4_epi32(__m128i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_extracti32x4_epi32
  // CHECK: @llvm.x86.avx512.mask.vextracti32x4
  return _mm512_mask_extracti32x4_epi32(__W, __U, __A, 3); 
}

__m128i test_mm512_maskz_extracti32x4_epi32(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_extracti32x4_epi32
  // CHECK: @llvm.x86.avx512.mask.vextracti32x4
  return _mm512_maskz_extracti32x4_epi32(__U, __A, 3); 
}

__m256i test_mm512_extracti64x4_epi64(__m512i __A) {
  // CHECK-LABEL: @test_mm512_extracti64x4_epi64
  // CHECK: @llvm.x86.avx512.mask.vextracti64x4
  return _mm512_extracti64x4_epi64(__A, 1); 
}

__m256i test_mm512_mask_extracti64x4_epi64(__m256i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_extracti64x4_epi64
  // CHECK: @llvm.x86.avx512.mask.vextracti64x4
  return _mm512_mask_extracti64x4_epi64(__W, __U, __A, 1); 
}

__m256i test_mm512_maskz_extracti64x4_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_extracti64x4_epi64
  // CHECK: @llvm.x86.avx512.mask.vextracti64x4
  return _mm512_maskz_extracti64x4_epi64(__U, __A, 1); 
}

__m512d test_mm512_insertf64x4(__m512d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm512_insertf64x4
  // CHECK: @llvm.x86.avx512.mask.insertf64x4
  return _mm512_insertf64x4(__A, __B, 1);
}

__m512d test_mm512_mask_insertf64x4(__m512d __W, __mmask8 __U, __m512d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm512_mask_insertf64x4
  // CHECK: @llvm.x86.avx512.mask.insertf64x4
  return _mm512_mask_insertf64x4(__W, __U, __A, __B, 1); 
}

__m512d test_mm512_maskz_insertf64x4(__mmask8 __U, __m512d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm512_maskz_insertf64x4
  // CHECK: @llvm.x86.avx512.mask.insertf64x4
  return _mm512_maskz_insertf64x4(__U, __A, __B, 1); 
}

__m512i test_mm512_inserti64x4(__m512i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm512_inserti64x4
  // CHECK: @llvm.x86.avx512.mask.inserti64x4
  return _mm512_inserti64x4(__A, __B, 1); 
}

__m512i test_mm512_mask_inserti64x4(__m512i __W, __mmask8 __U, __m512i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm512_mask_inserti64x4
  // CHECK: @llvm.x86.avx512.mask.inserti64x4
  return _mm512_mask_inserti64x4(__W, __U, __A, __B, 1); 
}

__m512i test_mm512_maskz_inserti64x4(__mmask8 __U, __m512i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm512_maskz_inserti64x4
  // CHECK: @llvm.x86.avx512.mask.inserti64x4
  return _mm512_maskz_inserti64x4(__U, __A, __B, 1); 
}

__m512d test_mm512_getmant_round_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_getmant_round_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_getmant_round_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_mask_getmant_round_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_getmant_round_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_mask_getmant_round_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_maskz_getmant_round_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_getmant_round_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_maskz_getmant_round_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_getmant_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_getmant_pd(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512d test_mm512_mask_getmant_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_mask_getmant_pd(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512d test_mm512_maskz_getmant_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_getmant_pd
  // CHECK: @llvm.x86.avx512.mask.getmant.pd.512
  return _mm512_maskz_getmant_pd(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512 test_mm512_getmant_round_ps(__m512 __A) {
  // CHECK-LABEL: @test_mm512_getmant_round_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_getmant_round_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_mask_getmant_round_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_getmant_round_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_mask_getmant_round_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_maskz_getmant_round_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_getmant_round_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_maskz_getmant_round_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_getmant_ps(__m512 __A) {
  // CHECK-LABEL: @test_mm512_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_getmant_ps(__A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512 test_mm512_mask_getmant_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_mask_getmant_ps(__W, __U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512 test_mm512_maskz_getmant_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_getmant_ps
  // CHECK: @llvm.x86.avx512.mask.getmant.ps.512
  return _mm512_maskz_getmant_ps(__U, __A,_MM_MANT_NORM_p5_2, _MM_MANT_SIGN_nan); 
}

__m512d test_mm512_getexp_round_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_getexp_round_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_getexp_round_pd(__A, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_mask_getexp_round_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_getexp_round_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_mask_getexp_round_pd(__W, __U, __A, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_maskz_getexp_round_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_getexp_round_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_maskz_getexp_round_pd(__U, __A, _MM_FROUND_CUR_DIRECTION); 
}

__m512d test_mm512_getexp_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_getexp_pd(__A); 
}

__m512d test_mm512_mask_getexp_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_mask_getexp_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_getexp_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_getexp_pd
  // CHECK: @llvm.x86.avx512.mask.getexp.pd.512
  return _mm512_maskz_getexp_pd(__U, __A); 
}

__m512 test_mm512_getexp_round_ps(__m512 __A) {
  // CHECK-LABEL: @test_mm512_getexp_round_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_getexp_round_ps(__A, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_mask_getexp_round_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_getexp_round_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_mask_getexp_round_ps(__W, __U, __A, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_maskz_getexp_round_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_getexp_round_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_maskz_getexp_round_ps(__U, __A, _MM_FROUND_CUR_DIRECTION); 
}

__m512 test_mm512_getexp_ps(__m512 __A) {
  // CHECK-LABEL: @test_mm512_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_getexp_ps(__A); 
}

__m512 test_mm512_mask_getexp_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_mask_getexp_ps(__W, __U, __A); 
}

__m512 test_mm512_maskz_getexp_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_getexp_ps
  // CHECK: @llvm.x86.avx512.mask.getexp.ps.512
  return _mm512_maskz_getexp_ps(__U, __A); 
}
