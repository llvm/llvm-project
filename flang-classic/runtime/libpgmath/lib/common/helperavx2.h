//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

//    Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#if CONFIG == 1

#ifndef __AVX2__
#error Please specify -mavx2.
#endif

#else
#error CONFIG macro invalid or not defined
#endif

#define ENABLE_DP
#define LOG2VECTLENDP 2
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

typedef __m256i vmask;
typedef __m256i vopmask;

typedef __m256d vdouble;
typedef __m128i vint;

typedef __m256 vfloat;
typedef __m256i vint2;

//

#ifndef __SLEEF_H__
void Sleef_x86CpuID(int32_t out[4], uint32_t eax, uint32_t ecx);
#endif

static int cpuSupportsAVX2() {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 7, 0);
    return (reg[1] & (1 << 5)) != 0;
}

static int cpuSupportsFMA() {
    int32_t reg[4];
    Sleef_x86CpuID(reg, 1, 0);
    return (reg[2] & (1 << 12)) != 0;
}

#if CONFIG == 1 && defined(__AVX2__)
static INLINE int vavailability_i(int name) {
  int d = cpuSupportsAVX2() && cpuSupportsFMA();
  return d ? 3 : 0;
}
#define ISANAME "AVX2"
#define DFTPRIORITY 25
#endif

static INLINE void vprefetch_v_p(const char *ptr) { _mm_prefetch(ptr, _MM_HINT_T0); }

static INLINE int vtestallones_i_vo32(vopmask g) {
  return _mm_test_all_ones(_mm_and_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1)));
}

static INLINE int vtestallones_i_vo64(vopmask g) {
  return _mm_test_all_ones(_mm_and_si128(_mm256_extractf128_si256(g, 0), _mm256_extractf128_si256(g, 1)));
}

//

static INLINE vdouble vcast_vd_d(double d) { return _mm256_set1_pd(d); }
static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return _mm256_castpd_si256(vd); }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return _mm256_castsi256_pd(vm);  }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return _mm256_castpd_si256(vd); }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return _mm256_castsi256_pd(vi); }

//

static vint2 vloadu_vi2_p(int32_t *p) { return _mm256_loadu_si256((__m256i const *)p); }
static void vstoreu_v_p_vi2(int32_t *p, vint2 v) { _mm256_storeu_si256((__m256i *)p, v); }
static vint vloadu_vi_p(int32_t *p) { return _mm_loadu_si128((__m128i *)p); }
static void vstoreu_v_p_vi(int32_t *p, vint v) { _mm_storeu_si128((__m128i *)p, v); }

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) { return vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) { return vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vxor_vm_vo64_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_and_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_or_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_xor_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }

static INLINE vopmask vcast_vo32_vo64(vopmask o) {
  return _mm256_permutevar8x32_epi32(o, _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0));
}

static INLINE vopmask vcast_vo64_vo32(vopmask o) {
  return _mm256_permutevar8x32_epi32(o, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
}

//

static INLINE vint vrint_vi_vd(vdouble vd) { return _mm256_cvtpd_epi32(vd); }
static INLINE vint vtruncate_vi_vd(vdouble vd) { return _mm256_cvttpd_epi32(vd); }
static INLINE vdouble vrint_vd_vd(vdouble vd) { return _mm256_round_pd(vd, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC); }
static INLINE vfloat vrint_vf_vf(vfloat vd) { return _mm256_round_ps(vd, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC); }
static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return _mm256_round_pd(vd, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC); }
static INLINE vfloat vtruncate_vf_vf(vfloat vf) { return _mm256_round_ps(vf, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC); }
static INLINE vdouble vcast_vd_vi(vint vi) { return _mm256_cvtepi32_pd(vi); }
static INLINE vint vcast_vi_i(int i) { return _mm_set1_epi32(i); }

static INLINE vint2 vcastu_vi2_vi(vint vi) {
  return _mm256_slli_epi64(_mm256_cvtepi32_epi64(vi), 32);
}

static INLINE vint vcastu_vi_vi2(vint2 vi) {
  return _mm_or_si128(_mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm256_castsi256_si128(vi)), _mm_set1_ps(0), 0x0d)),
  		      _mm_castps_si128(_mm_shuffle_ps(_mm_set1_ps(0), _mm_castsi128_ps(_mm256_extractf128_si256(vi, 1)), 0xd0)));
}

static INLINE vmask vcast_vm_i_i(int i0, int i1) {
  return _mm256_set_epi32(i0, i1, i0, i1, i0, i1, i0, i1);
}

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) { return _mm256_cmpeq_epi64(x, y); }
static INLINE vopmask vgt64_vo_vm_vm(vmask x, vmask y) { return _mm256_cmpgt_epi64(x, y); }

//

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return _mm256_add_pd(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return _mm256_sub_pd(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return _mm256_mul_pd(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return _mm256_div_pd(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return _mm256_div_pd(_mm256_set1_pd(1), x); }
static INLINE vdouble vsqrt_vd_vd(vdouble x) { return _mm256_sqrt_pd(x); }
static INLINE vdouble vabs_vd_vd(vdouble d) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), d); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return _mm256_xor_pd(_mm256_set1_pd(-0.0), d); }
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fmadd_pd(x, y, z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fmsub_pd(x, y, z); }
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fnmadd_pd(x, y, z); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return _mm256_max_pd(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return _mm256_min_pd(x, y); }

static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fmadd_pd(x, y, z); }
static INLINE vdouble vfmapp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fmadd_pd(x, y, z); }
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fmsub_pd(x, y, z); }
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fnmadd_pd(x, y, z); }
static INLINE vdouble vfmann_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return _mm256_fnmsub_pd(x, y, z); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_EQ_OQ)); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_NEQ_UQ)); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_LT_OQ)); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_LE_OQ)); }
static INLINE vopmask vnlt_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_NLT_UQ)); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_GT_OQ)); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return vreinterpret_vm_vd(_mm256_cmp_pd(x, y, _CMP_GE_OQ)); }

//

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return _mm_add_epi32(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return _mm_sub_epi32(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vsub_vi_vi_vi(vcast_vi_i(0), e); }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return _mm_and_si128(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return _mm_andnot_si128(x, y); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return _mm_or_si128(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return _mm_xor_si128(x, y); }

static INLINE vint vandnot_vi_vo_vi(vopmask m, vint y) { return _mm_andnot_si128(_mm256_castsi256_si128(m), y); }
static INLINE vint vand_vi_vo_vi(vopmask m, vint y) { return _mm_and_si128(_mm256_castsi256_si128(m), y); }

static INLINE vint vsll_vi_vi_i(vint x, int c) { return _mm_slli_epi32(x, c); }
static INLINE vint vsrl_vi_vi_i(vint x, int c) { return _mm_srli_epi32(x, c); }
static INLINE vint vsra_vi_vi_i(vint x, int c) { return _mm_srai_epi32(x, c); }

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return _mm_cmpeq_epi32(x, y); }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return _mm_cmpgt_epi32(x, y); }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) { return _mm256_castsi128_si256(_mm_cmpeq_epi32(x, y)); }
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) { return _mm256_castsi128_si256(_mm_cmpgt_epi32(x, y)); }

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) { return _mm_blendv_epi8(y, x, _mm256_castsi256_si128(m)); }

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask o, vdouble x, vdouble y) { return _mm256_blendv_pd(y, x, _mm256_castsi256_pd(o)); }
static INLINE vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0) { return _mm256_permutevar_pd(_mm256_set_pd(v1, v0, v1, v0), o); }

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  __m256i v = _mm256_castpd_si256(vsel_vd_vo_vd_vd(o0, _mm256_castsi256_pd(_mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0)),
						   vsel_vd_vo_vd_vd(o1, _mm256_castsi256_pd(_mm256_set_epi32(3, 2, 3, 2, 3, 2, 3, 2)),
								    vsel_vd_vo_vd_vd(o2, _mm256_castsi256_pd(_mm256_set_epi32(5, 4, 5, 4, 5, 4, 5, 4)),
										     _mm256_castsi256_pd(_mm256_set_epi32(7, 6, 7, 6, 7, 6, 7, 6))))));
  return _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(_mm256_set_pd(d3, d2, d1, d0)), v));
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o1, d0, d1, d2, d2);
}

static INLINE vopmask visinf_vo_vd(vdouble d) {
  return vreinterpret_vm_vd(_mm256_cmp_pd(vabs_vd_vd(d), _mm256_set1_pd(INFINITY), _CMP_EQ_OQ));
}

static INLINE vopmask vispinf_vo_vd(vdouble d) {
  return vreinterpret_vm_vd(_mm256_cmp_pd(d, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ));
}

static INLINE vopmask visminf_vo_vd(vdouble d) {
  return vreinterpret_vm_vd(_mm256_cmp_pd(d, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ));
}

static INLINE vopmask visnan_vo_vd(vdouble d) {
  return vreinterpret_vm_vd(_mm256_cmp_pd(d, d, _CMP_NEQ_UQ));
}

#if defined(_MSC_VER)
// This function is needed when debugging on MSVC.
static INLINE double vcast_d_vd(vdouble v) {
  double s[4];
  _mm256_storeu_pd(s, v);
  return s[0];
}
#endif

static INLINE vdouble vset_vd_d_d(const double hi, const double lo) { return _mm256_set_pd(hi, lo, hi, lo); }

static INLINE vdouble vload_vd_p(const double *ptr) { return _mm256_load_pd(ptr); }
static INLINE vdouble vloadu_vd_p(const double *ptr) { return _mm256_loadu_pd(ptr); }

static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { _mm256_store_pd(ptr, v); }
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) { _mm256_storeu_pd(ptr, v); }

//

static INLINE vint2 vcast_vi2_vm(vmask vm) { return vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }

static INLINE vint2 vrint_vi2_vf(vfloat vf) { return vcast_vi2_vm(_mm256_cvtps_epi32(vf)); }
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcast_vi2_vm(_mm256_cvttps_epi32(vf)); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return _mm256_cvtepi32_ps(vcast_vm_vi2(vi)); }
static INLINE vfloat vcast_vf_f(float f) { return _mm256_set1_ps(f); }
static INLINE vint2 vcast_vi2_i(int i) { return _mm256_set1_epi32(i); }
static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return _mm256_castps_si256(vf); }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return _mm256_castsi256_ps(vm); }

static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return vreinterpret_vf_vm(vcast_vm_vi2(vi)); }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return vcast_vi2_vm(vreinterpret_vm_vf(vf)); }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return _mm256_add_ps(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return _mm256_sub_ps(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return _mm256_mul_ps(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return _mm256_div_ps(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return vdiv_vf_vf_vf(vcast_vf_f(1.0f), x); }
static INLINE vfloat vsqrt_vf_vf(vfloat x) { return _mm256_sqrt_ps(x); }
static INLINE vfloat vabs_vf_vf(vfloat f) { return vreinterpret_vf_vm(vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f)), vreinterpret_vm_vf(f))); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f)), vreinterpret_vm_vf(d))); }
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fmadd_ps(x, y, z); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fmsub_ps(x, y, z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fnmadd_ps(x, y, z); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return _mm256_max_ps(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return _mm256_min_ps(x, y); }

static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fmadd_ps(x, y, z); }
static INLINE vfloat vfmapp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fmadd_ps(x, y, z); }
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fmsub_ps(x, y, z); }
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fnmadd_ps(x, y, z); }
static INLINE vfloat vfmann_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fnmsub_ps(x, y, z); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_EQ_OQ)); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_NEQ_UQ)); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LT_OQ)); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LE_OQ)); }
static INLINE vopmask vnlt_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_NLT_UQ)); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_GT_OQ)); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_GE_OQ)); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_add_epi32(x, y); }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_sub_epi32(x, y); }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vsub_vi2_vi2_vi2(vcast_vi2_i(0), e); }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_and_si256(x, y); }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_andnot_si256(x, y); }
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_or_si256(x, y); }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_xor_si256(x, y); }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return vand_vi2_vi2_vi2(vcast_vi2_vm(x), y); }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) { return vandnot_vi2_vi2_vi2(vcast_vi2_vm(x), y); }

static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return _mm256_slli_epi32(x, c); }
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return _mm256_srli_epi32(x, c); }
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return _mm256_srai_epi32(x, c); }

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpeq_epi32(x, y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpgt_epi32(x, y); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpeq_epi32(x, y); }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpgt_epi32(x, y); }

static INLINE vopmask veq64_vo_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpeq_epi64(x, y); }
static INLINE vopmask vgt64_vo_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpgt_epi64(x, y); }
static INLINE vint2 veq64_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpeq_epi64(x, y); }
static INLINE vint2 vgt64_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_cmpgt_epi64(x, y); }

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) {
  return _mm256_blendv_epi8(y, x, m);
}

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask o, vfloat x, vfloat y) { return _mm256_blendv_ps(y, x, _mm256_castsi256_ps(o)); }

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

#ifdef _MSC_VER
// This function is needed when debugging on MSVC.
static INLINE float vcast_f_vf(vfloat v) {
  float s[8];
  _mm256_storeu_ps(s, v);
  return s[0];
}
#endif

static INLINE vfloat vload_vf_p(const float *ptr) { return _mm256_load_ps(ptr); }
static INLINE vfloat vloadu_vf_p(const float *ptr) { return _mm256_loadu_ps(ptr); }

static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { _mm256_store_ps(ptr, v); }
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) { _mm256_storeu_ps(ptr, v); }

//

#define PNMASK ((vdouble) { +0.0, -0.0, +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0, -0.0, +0.0 })
#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f })

static INLINE vdouble vposneg_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(PNMASK))); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(NPMASK))); }
static INLINE vfloat vposneg_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(PNMASKf))); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(d), vreinterpret_vm_vf(NPMASKf))); }

static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return _mm256_addsub_pd(x, y); }
static INLINE vfloat vsubadd_vf_vf_vf(vfloat x, vfloat y) { return _mm256_addsub_ps(x, y); }

static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vmla_vd_vd_vd_vd(x, y, vnegpos_vd_vd(z)); }
static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vmla_vf_vf_vf_vf(x, y, vnegpos_vf_vf(z)); }

static INLINE vdouble vrev21_vd_vd(vdouble d0) { return  _mm256_shuffle_pd(d0, d0, (0 << 3) | (1 << 2) | (0 << 1) | (1 << 0)); }
static INLINE vdouble vreva2_vd_vd(vdouble d0) { d0 = _mm256_permute2f128_pd(d0, d0, 1); return _mm256_shuffle_pd(d0, d0, (1 << 3) | (0 << 2) | (1 << 1) | (0 << 0)); }

static INLINE void vstream_v_p_vd(double *ptr, vdouble v) { _mm256_stream_pd(ptr, v); }
static INLINE void vscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) {
  _mm_store_pd(&ptr[(offset + step * 0)*2], _mm256_extractf128_pd(v, 0));
  _mm_store_pd(&ptr[(offset + step * 1)*2], _mm256_extractf128_pd(v, 1));
}

static INLINE void vsscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) {
  _mm_stream_pd(&ptr[(offset + step * 0)*2], _mm256_extractf128_pd(v, 0));
  _mm_stream_pd(&ptr[(offset + step * 1)*2], _mm256_extractf128_pd(v, 1));
}

//

static INLINE vfloat vrev21_vf_vf(vfloat d0) { return _mm256_shuffle_ps(d0, d0, (2 << 6) | (3 << 4) | (0 << 2) | (1 << 0)); }
static INLINE vfloat vreva2_vf_vf(vfloat d0) { d0 = _mm256_permute2f128_ps(d0, d0, 1); return _mm256_shuffle_ps(d0, d0, (1 << 6) | (0 << 4) | (3 << 2) | (2 << 0)); }
static INLINE vfloat vmoveldup_vf_vf(vfloat d0) { return _mm256_moveldup_ps(d0); }
static INLINE vfloat vmovehdup_vf_vf(vfloat d0) { return _mm256_movehdup_ps(d0); }

static INLINE vdouble vmoveldup_vd_vd(vdouble d0) { return _mm256_movedup_pd(d0); }
static INLINE vdouble vmovehdup_vd_vd(vdouble d0) { return _mm256_unpackhi_pd(d0, d0); }

static INLINE void vstream_v_p_vf(float *ptr, vfloat v) { _mm256_stream_ps(ptr, v); }

static INLINE void vscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  _mm_storel_pd((double *)(ptr+(offset + step * 0)*2), _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 0))));
  _mm_storeh_pd((double *)(ptr+(offset + step * 1)*2), _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 0))));
  _mm_storel_pd((double *)(ptr+(offset + step * 2)*2), _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 1))));
  _mm_storeh_pd((double *)(ptr+(offset + step * 3)*2), _mm_castsi128_pd(_mm_castps_si128(_mm256_extractf128_ps(v, 1))));
}

static INLINE void vsscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) { vscatter2_v_p_i_i_vf(ptr, offset, step, v); }

static INLINE int vtestz_i_vf_vf(vfloat x, vfloat y) { return _mm256_testz_ps(x, y); }
static INLINE int vtestz_i_vo_vo(vopmask x, vopmask y) { return _mm256_testz_ps((__m256)x, (__m256)y); }

static INLINE int vtestz_i_vo(vopmask x) { return _mm256_testz_si256(x, x); }

static INLINE vint2 vsll64_vi2_vi2_i(vint2 x, int c) { return _mm256_slli_epi64(x, c); }
static INLINE vint2 vsrl64_vi2_vi2_i(vint2 x, int c) { return _mm256_srli_epi64(x, c); }

static INLINE vint2 vmulu_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_mul_epu32(x, y); };
static INLINE vint2 vadd64_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_add_epi64(x, y); }
static INLINE vint2 vsub64_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_sub_epi64(x, y); }


static INLINE vdouble vcvtsi64_vd_vi2(vint2 x) {
  union {
    vdouble vd;
    vint2 vi2;
    int64_t ti[sizeof(vdouble) / sizeof(int64_t)];
    double td[sizeof(vdouble) / sizeof(double)];
  } in, out;
  in.vi2 = x;
  for (unsigned i = 0; i < sizeof(vdouble) / sizeof(int64_t); i++)
    out.td[i] = (double)in.ti[i];
  return out.vd;
}

static INLINE vint vhi64_vi_vi2(vint2 x) { return (vint)_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(x, _mm256_set_epi32(0, 0, 0, 0, 7, 5, 3, 1))); }

#if (defined __AVX512VL__)
static INLINE vdouble vgetexp_vd_vd(vdouble d) { return _mm256_getexp_pd(d); }
static INLINE vfloat vgetexp_vf_vf(vfloat d) { return _mm256_getexp_ps(d); }
static INLINE vdouble vgetmant_vd_vd(vdouble d) { return _mm256_getmant_pd(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan); }
static INLINE vfloat vgetmant_vf_vf(vfloat d) { return _mm256_getmant_ps(d, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan); }
#endif
