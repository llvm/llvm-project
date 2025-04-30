//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

//    Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#if CONFIG == 1

#ifndef __AVX512F__
#error Please specify -mavx512f.
#endif

#else
#error CONFIG macro invalid or not defined
#endif


#if     defined(__knl) || defined (__knl__)
#define FCN_AVX512(a)                                                         \
        a##_knl
#else
#define FCN_AVX512(a)                                                         \
        a##_avx512
#endif

#define _STRINGIFY(_n) #_n
#define STRINGIFY(_n) _STRINGIFY(_n)

#define ENABLE_DP
#define LOG2VECTLENDP 3
#define VECTLENDP (1 << LOG2VECTLENDP)
#define ENABLE_FMA_DP

#define ENABLE_SP
#define LOG2VECTLENSP (LOG2VECTLENDP+1)
#define VECTLENSP (1 << LOG2VECTLENSP)
#define ENABLE_FMA_SP

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <stdint.h>
#include "misc.h"

typedef __m512i vmask;
typedef __mmask16 vopmask;

typedef __m512d vdouble;
typedef __m256i vint;

typedef __m512 vfloat;
typedef __m512i vint2;

//

#ifndef __SLEEF_H__
void Sleef_x86CpuID(int32_t out[4], uint32_t eax, uint32_t ecx);
#endif

static int cpuSupportsAVX512F() {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 7, 0);
    return (reg[1] & (1 << 16)) != 0;
}

#if CONFIG == 1 && defined(__AVX512F__)
static INLINE int vavailability_i(int name) {
  int d = cpuSupportsAVX512F();
  return d ? 3 : 0;
}
#define ISANAME "AVX512F"
#define DFTPRIORITY 30
#endif

static INLINE void vprefetch_v_p(const void *ptr) { _mm_prefetch((const char *)ptr, _MM_HINT_T0); }

#ifdef __INTEL_COMPILER
static INLINE int vtestallones_i_vo64(vopmask g) { return _mm512_mask2int(g) == 0xff; }
static INLINE int vtestallones_i_vo32(vopmask g) { return _mm512_mask2int(g) == 0xffff; }
#else
static INLINE int vtestallones_i_vo64(vopmask g) { return g == 0xff; }
static INLINE int vtestallones_i_vo32(vopmask g) { return g == 0xffff; }
#endif

//

static vint2 vloadu_vi2_p(int32_t *p) { return _mm512_loadu_si512((__m512i const *)p); }
static void vstoreu_v_p_vi2(int32_t *p, vint2 v) { return _mm512_storeu_si512((__m512i *)p, v); }
static vint vloadu_vi_p(int32_t *p) { return _mm256_loadu_si256((__m256i const *)p); }
static void vstoreu_v_p_vi(int32_t *p, vint v) { return _mm256_storeu_si256((__m256i *)p, v); }

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return _mm512_and_si512(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return _mm512_andnot_si512(x, y); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return _mm512_or_si512(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return _mm512_xor_si512(x, y); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kand(x, y); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kandn(x, y); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kor(x, y); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return _mm512_kxor(x, y); }

static INLINE vmask vand_vm_vo64_vm(vopmask o, vmask m) { return _mm512_mask_and_epi64(_mm512_set1_epi32(0), o, m, m); }
static INLINE vmask vandnot_vm_vo64_vm(vopmask o, vmask m) { return _mm512_mask_and_epi64(m, o, _mm512_set1_epi32(0), _mm512_set1_epi32(0)); }
static INLINE vmask vor_vm_vo64_vm(vopmask o, vmask m) { return _mm512_mask_or_epi64(m, o, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1)); }

static INLINE vmask vand_vm_vo32_vm(vopmask o, vmask m) { return _mm512_mask_and_epi32(_mm512_set1_epi32(0), o, m, m); }
static INLINE vmask vandnot_vm_vo32_vm(vopmask o, vmask m) { return _mm512_mask_and_epi32(m, o, _mm512_set1_epi32(0), _mm512_set1_epi32(0)); }
static INLINE vmask vor_vm_vo32_vm(vopmask o, vmask m) { return _mm512_mask_or_epi32(m, o, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1)); }

static INLINE vopmask vcast_vo32_vo64(vopmask o) { return o; }
static INLINE vopmask vcast_vo64_vo32(vopmask o) { return o; }

//

static INLINE vint vrint_vi_vd(vdouble vd) {
  return _mm512_cvt_roundpd_epi32(vd, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
}

static INLINE vint vtruncate_vi_vd(vdouble vd) {
  return _mm512_cvt_roundpd_epi32(vd, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
}

static INLINE vdouble vcast_vd_vi(vint vi) { return _mm512_cvtepi32_pd(vi); }
static INLINE vint vcast_vi_i(int i) { return _mm256_set1_epi32(i); }

static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return vcast_vd_vi(vtruncate_vi_vd(vd)); }
static INLINE vdouble vrint_vd_vd(vdouble vd) { return vcast_vd_vi(vrint_vi_vd(vd)); }

static INLINE vint2 vcastu_vi2_vi(vint vi) {
  return _mm512_maskz_permutexvar_epi32(0xaaaa, _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), _mm512_castsi256_si512(vi));
}

static INLINE vint vcastu_vi_vi2(vint2 vi) {
  return _mm512_castsi512_si256(_mm512_maskz_permutexvar_epi32(0x00ff, _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1), vi));
}

static INLINE vmask vcast_vm_i_i(int i0, int i1) { return _mm512_set_epi32(i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1, i0, i1); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) { return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_EQ); }

//

static INLINE vdouble vcast_vd_d(double d) { return _mm512_set1_pd(d); }
static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (__m512i)vd; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (__m512d)vm; }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return (__m512i)vd; }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return (__m512d)vi; }

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return _mm512_add_pd(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return _mm512_sub_pd(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return _mm512_mul_pd(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return _mm512_div_pd(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return _mm512_div_pd(_mm512_set1_pd(1), x); }
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return _mm512_sqrt_pd(x); }
static INLINE vdouble vabs_vd_vd(vdouble d) { return (__m512d)_mm512_andnot_si512((__m512i)_mm512_set1_pd(-0.0), (__m512i)d); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return (__m512d)_mm512_xor_si512((__m512i)_mm512_set1_pd(-0.0), (__m512i)d); }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmadd_pd(x, y, z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmsub_pd(x, y, z); }
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fnmadd_pd(x, y, z); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return _mm512_max_pd(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return _mm512_min_pd(x, y); }

static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmadd_pd(x, y, z); }
static INLINE vdouble vfmapp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmadd_pd(x, y, z); }
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmsub_pd(x, y, z); }
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fnmadd_pd(x, y, z); }
static INLINE vdouble vfmann_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fnmsub_pd(x, y, z); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OQ); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_LE_OQ); }
static INLINE vopmask vnlt_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_NLT_UQ); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_GT_OQ); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return _mm512_cmp_pd_mask(x, y, _CMP_GE_OQ); }

//

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return _mm256_add_epi32(x, y); }
static INLINE vint vadd64_vi_vi_vi(vint x, vint y) { return _mm256_add_epi64(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return _mm256_sub_epi32(x, y); }
static INLINE vint vsub64_vi_vi_vi(vint x, vint y) { return _mm256_sub_epi64(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vsub_vi_vi_vi(vcast_vi_i(0), e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return _mm256_and_si256(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return _mm256_andnot_si256(x, y); }

static INLINE vint vandnot_vi_vo_vi(vopmask o, vint y) {
  return _mm512_castsi512_si256(_mm512_mask_and_epi32(_mm512_castsi256_si512(y), o, _mm512_set1_epi32(0), _mm512_set1_epi32(0)));
}
static INLINE vint vand_vi_vo_vi(vopmask o, vint y) {
  return _mm512_castsi512_si256(_mm512_mask_and_epi32(_mm512_set1_epi32(0), o, _mm512_castsi256_si512(y), _mm512_castsi256_si512(y)));
}

static INLINE vint vor_vi_vi_vi(vint x, vint y) { return _mm256_or_si256(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return _mm256_xor_si256(x, y); }
static INLINE vint vsll_vi_vi_i(vint x, int c) { return _mm256_slli_epi32(x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return _mm256_srli_epi32(x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return _mm256_srai_epi32(x, c); }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return _mm256_cmpeq_epi32(x, y); }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return _mm256_cmpgt_epi32(x, y); }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) {
  return _mm512_cmp_epi32_mask(_mm512_castsi256_si512(x), _mm512_castsi256_si512(y), _MM_CMPINT_EQ);
}
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) {
  return _mm512_cmp_epi32_mask(_mm512_castsi256_si512(y), _mm512_castsi256_si512(x), _MM_CMPINT_LT);
}

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask mask, vdouble x, vdouble y) {
  return _mm512_mask_blend_pd(mask, y, x);
}

static INLINE CONST vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0) {
  return vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0));
}

// Probably this is faster
static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  __m512i v = _mm512_castpd_si512(vsel_vd_vo_vd_vd(o0, _mm512_castsi512_pd(_mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0)),
						   vsel_vd_vo_vd_vd(o1, _mm512_castsi512_pd(_mm512_set_epi64(1, 1, 1, 1, 1, 1, 1, 1)),
								    vsel_vd_vo_vd_vd(o2, _mm512_castsi512_pd(_mm512_set_epi64(2, 2, 2, 2, 2, 2, 2, 2)),
										     _mm512_castsi512_pd(_mm512_set_epi64(3, 3, 3, 3, 3, 3, 3, 3))))));
  return _mm512_permutexvar_pd(v, _mm512_castpd256_pd512(_mm256_set_pd(d3, d2, d1, d0)));
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o1, d0, d1, d2, d2);
}

static INLINE vopmask visinf_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(vabs_vd_vd(d), _mm512_set1_pd(INFINITY), _CMP_EQ_OQ);
}

static INLINE vopmask vispinf_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(d, _mm512_set1_pd(INFINITY), _CMP_EQ_OQ);
}

static INLINE vopmask visminf_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(d, _mm512_set1_pd(-INFINITY), _CMP_EQ_OQ);
}

static INLINE vopmask visnan_vo_vd(vdouble d) {
  return _mm512_cmp_pd_mask(d, d, _CMP_NEQ_UQ);
}

static INLINE vint vilogbk_vi_vd(vdouble d) { return vrint_vi_vd(_mm512_getexp_pd(d)); }

// vilogb2k_vi_vd is similar to vilogbk_vi_vd, but the argument has to
// be a normalized FP value.
static INLINE vint vilogb2k_vi_vd(vdouble d) { return vrint_vi_vd(_mm512_getexp_pd(d)); }

static INLINE vdouble vgetexp_vd_vd(vdouble d) { return _mm512_getexp_pd(d); }
static INLINE vfloat vgetexp_vf_vf(vfloat d) { return _mm512_getexp_ps(d); }

static INLINE vdouble vgetmant_vd_vd(vdouble d) { return _mm512_getmant_pd(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan); }
static INLINE vfloat vgetmant_vf_vf(vfloat d) { return _mm512_getmant_ps(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan); }

#if defined(__clang__)
#define vfixup_vd_vd_vd_vi2_i(a, b, c, imm) ({ _mm512_fixupimm_pd((a), (b), (c), (imm)); })
#define vfixup_vf_vf_vf_vi2_i(a, b, c, imm) ({ _mm512_fixupimm_ps((a), (b), (c), (imm)); })
#else
static INLINE vdouble vfixup_vd_vd_vd_vi2_i(vdouble a, vdouble b, vint2 c, int imm) { return _mm512_fixupimm_pd(a, b, c, imm); }
static INLINE vfloat vfixup_vf_vf_vf_vi2_i(vfloat a, vfloat b, vint2 c, int imm) { return _mm512_fixupimm_ps(a, b, c, imm); }
#endif

#if defined(_MSC_VER)
// This function is needed when debugging on MSVC.
static INLINE double vcast_d_vd(vdouble v) {
  double s[VECTLENDP];
  _mm512_storeu_pd(s, v);
  return s[0];
}
#endif

static INLINE vdouble vset_vd_d_d(const double hi, const double lo) { return _mm512_set_pd(hi, lo, hi, lo, hi, lo, hi, lo); }

static INLINE vdouble vload_vd_p(const double *ptr) { return _mm512_load_pd(ptr); }
static INLINE vdouble vloadu_vd_p(const double *ptr) { return _mm512_loadu_pd(ptr); }

static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { _mm512_store_pd(ptr, v); }
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) { _mm512_storeu_pd(ptr, v); }

//

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) {
  return _mm512_castsi512_si256(_mm512_mask_blend_epi32(m, _mm512_castsi256_si512(y), _mm512_castsi256_si512(x)));
}

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }

static INLINE vfloat vcast_vf_vi2(vint2 vi) { return _mm512_cvtepi32_ps((vmask)vcast_vm_vi2(vi)); }
static INLINE vfloat vcast_vf_f(float f) { return _mm512_set1_ps(f); }
static INLINE vint2 vcast_vi2_i(int i) { return _mm512_set1_epi32(i); }
static INLINE vint2 vrint_vi2_vf(vfloat vf) { return vcast_vi2_vm((vmask)_mm512_cvtps_epi32(vf)); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcast_vi2_vm((vmask)_mm512_cvttps_epi32(vf)); }
static INLINE vfloat vtruncate_vf_vf(vfloat vd) { return vcast_vf_vi2(vtruncate_vi2_vf(vd)); }
static INLINE vfloat vrint_vf_vf(vfloat vd) { return vcast_vf_vi2(vrint_vi2_vf(vd)); }

static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (vmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (vfloat)vm; }

static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return (vfloat)vi; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (vint2)vf; }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return _mm512_add_ps(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return _mm512_sub_ps(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return _mm512_mul_ps(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return _mm512_div_ps(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return vdiv_vf_vf_vf(vcast_vf_f(1.0f), x); }
static INLINE vfloat vsqrt_vf_vf(vfloat x) { return _mm512_sqrt_ps(x); }
static INLINE vfloat vabs_vf_vf(vfloat f) { return (vfloat)vandnot_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)f); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)vcast_vf_f(-0.0f), (vmask)d); }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmadd_ps(x, y, z); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmsub_ps(x, y, z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fnmadd_ps(x, y, z); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return _mm512_max_ps(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return _mm512_min_ps(x, y); }

static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmadd_ps(x, y, z); }
static INLINE vfloat vfmapp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmadd_ps(x, y, z); }
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmsub_ps(x, y, z); }
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fnmadd_ps(x, y, z); }
static INLINE vfloat vfmann_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fnmsub_ps(x, y, z); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ); }
static INLINE vopmask vnlt_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_NLT_UQ); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_add_epi32(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_sub_epi32(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vsub_vi2_vi2_vi2(vcast_vi2_i(0), e); }
static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_and_si512(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_andnot_si512(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_or_si512(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_xor_si512(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask o, vint2 m) {
  return _mm512_mask_and_epi32(_mm512_set1_epi32(0), o, m, m);
}

static INLINE vint2 vandnot_vi2_vo_vi2(vopmask o, vint2 m) {
  return _mm512_mask_and_epi32(m, o, _mm512_set1_epi32(0), _mm512_set1_epi32(0));
}

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return _mm512_slli_epi32(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return _mm512_srli_epi32(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return _mm512_srai_epi32(x, c); }
static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return _mm512_cmpeq_epi32_mask(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return _mm512_cmpgt_epi32_mask(x, y); }

static INLINE vopmask vgt64_vo_vm_vm(vmask x, vmask y) { return _mm512_cmpgt_epi64_mask(x, y); }
static INLINE vopmask veq64_vo_vi2_vi2(vint2 x, vint2 y) { return _mm512_cmpeq_epi64_mask(x, y); }
static INLINE vopmask vgt64_vo_vi2_vi2(vint2 x, vint2 y) { return _mm512_cmpgt_epi64_mask(x, y); }


static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) {
  __mmask16 m = _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_EQ);
  return _mm512_mask_and_epi32(_mm512_set1_epi32(0), m, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1));
}
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) {
  __mmask16 m = _mm512_cmp_epi32_mask(y, x, _MM_CMPINT_LT);
  return _mm512_mask_and_epi32(_mm512_set1_epi32(0), m, _mm512_set1_epi32(-1), _mm512_set1_epi32(-1));
}

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) {
  return _mm512_mask_blend_epi32(m, y, x);
}

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask m, vfloat x, vfloat y) {
  return _mm512_mask_blend_ps(m, y, x);
}

// At this point, the following three functions are implemented in a generic way,
// but I will try target-specific optimization later on.
static INLINE CONST vfloat vsel_vf_vo_f_f(vopmask o, float v1, float v0) {
  return vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0));
}

static INLINE vfloat vsel_vf_vo_vo_f_f_f(vopmask o0, vopmask o1, float d0, float d1, float d2) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2));
}

static INLINE vfloat vsel_vf_vo_vo_vo_f_f_f_f(vopmask o0, vopmask o1, vopmask o2, float d0, float d1, float d2, float d3) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3)));
}

static INLINE vopmask visinf_vo_vf(vfloat d) { return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(-INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return vneq_vo_vf_vf(d, d); }

static INLINE vint2 vilogbk_vi2_vf(vfloat d) { return vrint_vi2_vf(_mm512_getexp_ps(d)); }

#ifdef _MSC_VER
// This function is needed when debugging on MSVC.
static INLINE float vcast_f_vf(vfloat v) {
  float s[VECTLENSP];
  _mm512_storeu_ps(s, v);
  return s[0];
}
#endif

static INLINE vfloat vload_vf_p(const float *ptr) { return _mm512_load_ps(ptr); }
static INLINE vfloat vloadu_vf_p(const float *ptr) { return _mm512_loadu_ps(ptr); }

static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { _mm512_store_ps(ptr, v); }
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) { _mm512_storeu_ps(ptr, v); }

//

#define PNMASK ((vdouble) { +0.0, -0.0, +0.0, -0.0, +0.0, -0.0, +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0, -0.0, +0.0, -0.0, +0.0, -0.0, +0.0 })
#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f })

static INLINE vdouble vposneg_vd_vd(vdouble d) { return (vdouble)vxor_vm_vm_vm((vmask)d, (vmask)PNMASK); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return (vdouble)vxor_vm_vm_vm((vmask)d, (vmask)NPMASK); }
static INLINE vfloat vposneg_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)d, (vmask)PNMASKf); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)d, (vmask)NPMASKf); }

static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return vadd_vd_vd_vd(x, vnegpos_vd_vd(y)); }
static INLINE vfloat vsubadd_vf_vf_vf(vfloat x, vfloat y) { return vadd_vf_vf_vf(x, vnegpos_vf_vf(y)); }

static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm512_fmaddsub_pd(x, y, z); }
static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm512_fmaddsub_ps(x, y, z); }

static INLINE vdouble vrev21_vd_vd(vdouble vd) {
  return (vdouble)_mm512_permutexvar_epi32(_mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2), (__m512i)vd);
}

static INLINE vdouble vreva2_vd_vd(vdouble vd) {
  return (vdouble)_mm512_permutexvar_epi32(_mm512_set_epi32(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12), (__m512i)vd);
}

static INLINE void vstream_v_p_vd(double *ptr, vdouble v) { _mm512_stream_pd(ptr, v); }

static INLINE void vscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) {
  _mm_store_pd(&ptr[(offset + step * 0)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 0));
  _mm_store_pd(&ptr[(offset + step * 1)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 1));
  _mm_store_pd(&ptr[(offset + step * 2)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 2));
  _mm_store_pd(&ptr[(offset + step * 3)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 3));
}

static INLINE void vsscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) {
  _mm_stream_pd(&ptr[(offset + step * 0)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 0));
  _mm_stream_pd(&ptr[(offset + step * 1)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 1));
  _mm_stream_pd(&ptr[(offset + step * 2)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 2));
  _mm_stream_pd(&ptr[(offset + step * 3)*2], (__m128d)_mm512_extractf32x4_ps((__m512)v, 3));
}

//

static INLINE vfloat vrev21_vf_vf(vfloat vf) {
  return (vfloat)_mm512_permutexvar_epi32(_mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1), (__m512i)vf);
}

static INLINE vfloat vreva2_vf_vf(vfloat vf) {
  return (vfloat)_mm512_permutexvar_epi32(_mm512_set_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14), (__m512i)vf);
}

static INLINE vdouble vmoveldup_vd_vd(vdouble d0) { return _mm512_movedup_pd(d0); }
static INLINE vdouble vmovehdup_vd_vd(vdouble d0) { return _mm512_unpackhi_pd(d0, d0); }

static INLINE vfloat vmoveldup_vf_vf(vfloat d0) { return _mm512_moveldup_ps(d0); }
static INLINE vfloat vmovehdup_vf_vf(vfloat d0) { return _mm512_movehdup_ps(d0); }

static INLINE void vstream_v_p_vf(float *ptr, vfloat v) { _mm512_stream_ps(ptr, v); }

static INLINE void vscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  _mm_storel_pd((double *)(ptr+(offset + step * 0)*2), (__m128d)_mm512_extractf32x4_ps(v, 0));
  _mm_storeh_pd((double *)(ptr+(offset + step * 1)*2), (__m128d)_mm512_extractf32x4_ps(v, 0));
  _mm_storel_pd((double *)(ptr+(offset + step * 2)*2), (__m128d)_mm512_extractf32x4_ps(v, 1));
  _mm_storeh_pd((double *)(ptr+(offset + step * 3)*2), (__m128d)_mm512_extractf32x4_ps(v, 1));
  _mm_storel_pd((double *)(ptr+(offset + step * 4)*2), (__m128d)_mm512_extractf32x4_ps(v, 2));
  _mm_storeh_pd((double *)(ptr+(offset + step * 5)*2), (__m128d)_mm512_extractf32x4_ps(v, 2));
  _mm_storel_pd((double *)(ptr+(offset + step * 6)*2), (__m128d)_mm512_extractf32x4_ps(v, 3));
  _mm_storeh_pd((double *)(ptr+(offset + step * 7)*2), (__m128d)_mm512_extractf32x4_ps(v, 3));
}

static INLINE void vsscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) { vscatter2_v_p_i_i_vf(ptr, offset, step, v); }

static INLINE int vtestz_i_vo_vo(vopmask x, vopmask y) { return _mm512_kortestz(x, y); }
static INLINE int vtestz_i_vo(vopmask x) { return _mm512_kortestz(x, x); }

static INLINE vint2 vsll64_vi2_vi2_i(vint2 x, int c) { return _mm512_slli_epi64(x, c); }
static INLINE vint2 vsrl64_vi2_vi2_i(vint2 x, int c) { return _mm512_srli_epi64(x, c); }

static INLINE vint2 vmulu_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_mul_epu32(x, y); };
static INLINE vint2 vadd64_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_add_epi64(x, y); }
static INLINE vint2 vsub64_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm512_sub_epi64(x, y); }

static INLINE vint vhi64_vi_vi2(vint2 x) { return (vint)_mm512_castsi512_si256(_mm512_permutexvar_epi32(_mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1), x)); }

