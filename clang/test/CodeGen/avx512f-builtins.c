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
  return _mm512_cmp_round_ps_mask(a, b, 0, _MM_FROUND_TO_NEAREST_INT);
}

__mmask16 test_mm512_mask_cmp_round_ps_mask(__mmask16 m, __m512 a, __m512 b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_ps_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.ps.512
  return _mm512_mask_cmp_round_ps_mask(m, a, b, 0, _MM_FROUND_TO_NEAREST_INT);
}

__mmask16 test_mm512_cmp_ps_mask(__m512 a, __m512 b) {
  // check-label: @test_mm512_cmp_ps_mask
  // check: @llvm.x86.avx512.mask.cmp.ps.512
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
  return _mm512_cmp_round_pd_mask(a, b, 0, _MM_FROUND_TO_NEAREST_INT);
}

__mmask8 test_mm512_mask_cmp_round_pd_mask(__mmask8 m, __m512d a, __m512d b) {
  // CHECK-LABEL: @test_mm512_mask_cmp_round_pd_mask
  // CHECK: @llvm.x86.avx512.mask.cmp.pd.512
  return _mm512_mask_cmp_round_pd_mask(m, a, b, 0, _MM_FROUND_TO_NEAREST_INT);
}

__mmask8 test_mm512_cmp_pd_mask(__m512d a, __m512d b) {
  // check-label: @test_mm512_cmp_pd_mask
  // check: @llvm.x86.avx512.mask.cmp.pd.512
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
