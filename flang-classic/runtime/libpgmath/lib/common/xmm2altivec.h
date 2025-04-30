/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * CPP and some helper C functions to convert SSE/AVX Intel/GNU intrinsics to
 * IBM Altivec equivalents.
 * 
 * Not all intrinsics translate as single Altivec instruction, but most
 * do.  For an example, see "FMA Instructions" below.
 */

/*
 * XXX XXX XXX
 * WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
 *
 * Some of the CPP marcos used for scalar operations are not correct with
 * regards to elements other than 0 (first in the vector).
 *
 * The macros below work with the FMA versions of the math intrinsics
 * because the scalar versions of the intrinsics copied the vector version
 * with the addition of a broadcast/propagation of the scalar argument
 * (element 0) in to vector of the same precision.
 *
 * WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
 * XXX XXX XXX
 */


#if defined(TARGET_LINUX_POWER)
#include <altivec.h>
#include <assert.h>

typedef vector float __m128;
typedef vector double __m128d;
typedef vector VINT int __m128i;

/*
 * No corresponding Altivec intrinsic to generate a scalar mask
 * from corresponding vector elements.
 */
static inline unsigned int
_mm_movemask_epi8(__m128i a)
{
  unsigned char t[16] __attribute__((aligned(16)));
  unsigned int r;
  int i;

  vec_st((vector unsigned char)a, 0, t);
  r = 0;
  for (i = 0; i < 16; i++) {
    r = (r << 1) | (t[i] >> 7);
  }
  return r;
}

static inline unsigned int
_mm_movemask_epi32(__m128i a)
{
  unsigned int t[4] __attribute__((aligned(16)));
  unsigned int r;
  int i;

  vec_st((vector unsigned int)a, 0, t);
  r = 0;
  for (i = 0; i < 4; i++) {
    r = (r << 1) | (t[i] >> 31);
  }
  return r;
}

static inline __m128i
_mm_blend_epi32 (__m128i a, __m128i b, int imm8)
{
  unsigned int t[4] __attribute__((aligned(16)));
  int i;

  vec_st((vector unsigned int) a, 0, t);
  for (i = 0; i < 3; i++) {
    if (imm8 & 0x1)
      t[i] = vec_extract((vector unsigned int)b, i);
    imm8 >>= 1;
  }
  return vec_ld(0, (__m128i *)t);
}

static inline __m128
_mm_setr_ps (float e3, float e2, float e1, float e0)
{
  __m128 e = {e3, e2, e1, e0};
  return e;
}

static inline __m128d
_mm_setr_pd (double e1, double e0)
{
  __m128d e = {e1, e0};
  return e;
}

static inline __m128d
_mm_shuffle_pd(__m128d a, __m128d b, int imm8)
{
  double r[2];
  r[0] = imm8 & 0x1 ? vec_extract(a, 1) : vec_extract(a, 0);
  r[1] = imm8 & 0x2 ? vec_extract(b, 1) : vec_extract(b, 0);

  return vec_ld(0, (__m128d *)r);
}

/*
 * Quick way to determine whether any element in a vector mask
 * register is set.
 * 
 * No corresponding Altivec intrinsic.
 */ 
static inline unsigned int
_vec_any_nz(__m128i a)
{
  return vec_any_ne(a, (typeof(a))vec_splats(0));
}

static inline __m128d
_mm_cvtepi32_pd(__m128i a)
{
  //asm("xvcvsxwdp 34,34");
  __m128d r;
  r = vec_insert(1.0*vec_extract(a,0), r, 0);
  r = vec_insert(1.0*vec_extract(a,2), r, 1);
  return r;
}

static inline __m128d
_mm_min_sd (__m128d a, __m128d b)
{
  double aa = vec_extract(a, 0);
  double bb = vec_extract(b, 0);
  aa < bb ? aa : bb;
  return vec_insert(aa, a, 0);
}

static inline __m128d
_mm_max_sd (__m128d a, __m128d b)
{
  double aa = vec_extract(a, 0);
  double bb = vec_extract(b, 0);
  aa > bb ? aa : bb;
  return vec_insert(aa, a, 0);
}


/*
 * Logical
 */

#define	_mm_andnot_ps(_v,_w) vec_andc(_w,_v)     // different oder of arguments
#define	_mm_andnot_pd(_v,_w) vec_andc(_w,_v)     // different oder of arguments
#define	_mm_and_ps(_v,_w) vec_and(_v,_w)
#define	_mm_and_pd(_v,_w) vec_and(_v,_w)
#define	_mm_and_si128(_v,_w) vec_and(_v,_w)
#define	_mm_andnot_si128(_v,_w) vec_andc(_w,_v)  // different order of arguments
#define	_mm_or_ps(_v,_w) vec_or(_v,_w)
#define	_mm_or_pd(_v,_w) vec_or(_v,_w)
#define	_mm_or_si128(_v,_w) vec_or(_v,_w)
#define	_mm_xor_ps(_v,_w) vec_xor(_v,_w)
#define	_mm_xor_pd(_v,_w) vec_xor(_v,_w)
#define	_mm_xor_si128(_v,_w) vec_xor(_v,_w)

/*
 * Broadcast
 */

#define	_mm_set1_epi32(_v) (__m128i)vec_splats((int)_v)
#define	_mm_set1_epi64x(_v) (__m128i)vec_splats((long int)_v)
#define	_mm_set1_ps(_v) (__m128)vec_splats((float)_v)
#define	_mm_set1_pd(_v) (__m128d)vec_splats((double)_v)
//#define	_mm_setr_ps(_e,_f) (__m128d)vec_insert(_e, (__m128d)vec_splats(_f), 0)
//#define	_mm_setr_pd(_e,_f) (__m128d)vec_insert(_e, (__m128d)vec_splats(_f), 0)
#define	_mm_setzero_ps() (__m128)vec_splats((float)0.0)
#define	_mm_setzero_pd() (__m128d)vec_splats((double)0.0)

#define	_mm_cvtps_epi32(_v) vec_cts(_v,0)
// Need inline version #define	_mm_cvtepi32_pd(_v) vec_ctd(_v,0)
#define	_mm_cvtepi32_ps(_v) vec_ctf(_v,0)
#define	_mm_cvtss_f32(_v) (float)vec_extract(_v,0)
#define	_mm_cvtsd_f64(_v) (double)vec_extract(_v,0)
//#define	_mm_cvtpd_ps(_v) (__m128)vec_cvf(_v)	// Does not work
#define	_mm_cvtpd_ps(_v) vec_insert((float)vec_extract(_v,1), (vec_insert((float)vec_extract(_v,0), (__m128)vec_splats((float)0.0), 0)), 1)
#define	_mm_cvtss_sd(_v,_w) vec_insert((double)vec_extract(_w, 0), _v, 0)
#define _mm_extract_ps(_v,_i) vec_extract((vector int)_v,_i)

/*
 * Floating point
 */

#define	_mm_add_ps(_v,_w) vec_add(_v,_w)
#define	_mm_add_pd(_v,_w) vec_add(_v,_w)
#define	_mm_add_epi64(_v,_w) vec_add(_v,_w)
#define	_mm_mul_ps(_v,_w) vec_mul(_v,_w)
#define	_mm_mul_pd(_v,_w) vec_mul(_v,_w)
#define	_mm_sub_ps(_v,_w) vec_sub(_v,_w)
#define	_mm_sub_pd(_v,_w) vec_sub(_v,_w)
#define	_mm_sub_epi32(_v,_w) vec_sub(_v,_w)
#define	_mm_sub_epi64(_v,_w) vec_sub(_v,_w)
#define	_mm_div_ps(_v,_w) vec_div(_v,_w)
#define	_mm_div_pd(_v,_w) vec_div(_v,_w)
#define	_mm_sqrt_ps(_v) vec_sqrt(_v)
#define	_mm_sqrt_pd(_v) vec_sqrt(_v)

#define	_mm_add_ss(_s,_t) (_s+_t)
#define	_mm_add_sd(_s,_t) (_s+_t)
#define	_mm_mul_ss(_s,_t) (_s*_t)
#define	_mm_mul_sd(_s,_t) (_s*_t)
#define	_mm_sub_ss(_s,_t) (_s-_t)
#define	_mm_sub_sd(_s,_t) (_s-_t)
#define	_mm_div_ss(_s,_t) (_s/_t)
#define	_mm_div_sd(_s,_t) (_s/_t)

#define	_mm_floor_ps(_v) vec_floor(_v)
#define	_mm_floor_pd(_v) vec_floor(_v)

/*
 * FMA instructions.
 *
 * _mm_fnmadd_p{s,d} not the same as Altivec intrinsic vec_nmadd(a,b,c).
 * Altivec returns: -(a*b+c).
 * We want: (-(a*b)+c)
 */

#define	_mm_fmadd_ps(_v,_w,_x) vec_madd(_v,_w,_x)
#define	_mm_fmadd_pd(_v,_w,_x) vec_madd(_v,_w,_x)
#define	_mm_fmsub_ps(_v,_w,_x) vec_msub(_v,_w,_x)
#define	_mm_fmsub_pd(_v,_w,_x) vec_msub(_v,_w,_x)
#define	_mm_fnmadd_ps(_v,_w,_x) vec_madd((-(_v)),_w,_x)
#define	_mm_fnmadd_pd(_v,_w,_x) vec_madd((-(_v)),_w,_x)
#define	_mm_min_epi32(_v,_w) vec_min(_v,_w)
#define	_mm_max_epi32(_v,_w) vec_max(_v,_w)
#define	_mm_max_epu32(_v,_w) vec_max(_v,_w)
//#define	_mm_min_sd(_v,_w)

#define	_mm_fmadd_ss(_v,_w,_x) vec_madd(_v,_w,_x)//fmaf(_v,_w,_x) //((_v*_w)+_x)
#define	_mm_fmadd_sd(_v,_w,_x) vec_madd(_v,_w,_x)//fmaf(_v,_w,_x) //((_v*_w)+_x)
#define	_mm_fmsub_ss(_v,_w,_x) vec_msub(_v,_w,_x)//fmsf(_v,_w,_x) //((_v*_w)-_x)
#define	_mm_fmsub_sd(_v,_w,_x) vec_msub(_v,_w,_x)//fmsf(_v,_w,_x) //((_v*_w)-_x)

/*
 * Integer.
 */

#define	_mm_add_epi32(_v,_w) vec_add(_v,_w)
#define	_mm_sub_epi32(_v,_w) vec_sub(_v,_w)

/*
 * Merge.
 */

#define	_mm_blendv_ps(_v,_w,_m) vec_sel(_v,_w,_m)
#define	_mm_blendv_pd(_v,_w,_m) vec_sel(_v,_w,_m)

/*
 * Miscelaneous:
 * Vector op constant
 * Casting
 */

#define	_mm_castps_si128(_v) (__m128i)(_v)
#define	_mm_castpd_si128(_v) (__m128i)(_v)
#define	_mm_slli_epi32(_v,_c) vec_sl(_v,vec_splats((unsigned int)_c))
#define	_mm_slli_epi64(_v,_c) (__m128i)vec_sl((vector unsigned long)_v,vec_splats((unsigned long)_c))
#define	_mm_sllv_epi64(_v,_w) vec_sl((__m128i)_v,(vector unsigned long)_w)
#define	_mm_srli_epi32(_v,_c) vec_sr(_v,vec_splats((unsigned int)_c))
#define	_mm_srli_epi64(_v,_c) vec_sr(_v,vec_splats((unsigned long)_c))

/*
 * Comparision.
 *
 * The following 4 macros stole shamelessly from:
 * https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
 */

#define	_CAT(_a,_b,...) _a##_b
#define	_EMPTY()
#define	_DEFER(id) id _EMPTY()
#define	_EXPAND1(...) __VA_ARGS__
#define	_EXPAND(...) _EXPAND1(_EXPAND1(__VA_ARGS__))

#define	__CMP_EQ_OQ(_v,_w) (typeof(_v))vec_cmpeq(_v,_w)
#define	__CMP_EQ_OS(_v,_w) (typeof(_v))vec_cmpeq(_v,_w)
#define	__CMP_LE_OQ(_v,_w) (typeof(_v))vec_cmple(_v,_w)
#define	__CMP_LT_OS(_v,_w) (typeof(_v))vec_cmplt(_v,_w)
#define	__CMP_LT_OQ(_v,_w) (typeof(_v))vec_cmplt(_v,_w)
#define	__CMP_GE_OS(_v,_w) (typeof(_v))vec_cmpge(_v,_w)
#define	__CMP_GT_OS(_v,_w) (typeof(_v))vec_cmpgt(_v,_w)
#define	__CMP_GT_OQ(_v,_w) (typeof(_v))vec_cmpgt(_v,_w)
//#define	__CMP_NEQ_UQ(_v,_w) (typeof(_v))vec_andc((__m128i)vec_splats(0xffffffff),(__m128i)vec_cmpeq(_v, _w))
#define	__CMP_NEQ_UQ(_v,_w) (typeof(_v))vec_andc((__m128i)vec_splats(-1),(__m128i)vec_cmpeq(_v, _w))
#define	__CMP_NLT_UQ(_v,_w) (typeof(_v))vec_andc((__m128i)vec_splats(-1),(__m128i)vec_cmplt(_v, _w))
#define	__CMP_NGE_UQ(_v,_w) (typeof(_v))vec_andc((__m128i)vec_splats(-1),(__m128i)vec_cmpge(_v, _w))

#define	_mm_cmpeq_epi32(_v,_w) (__m128i)vec_cmpeq(_v,_w)
#define	_mm_cmpeq_epi64(_v,_w) (__m128i)vec_cmpeq(_v,_w)
#define	_mm_cmpgt_epi32(_v,_w) (__m128i)vec_cmpgt(_v,_w)
#define	_mm_cmpgt_epi64(_v,_w) (__m128i)vec_cmpgt(_v,_w)
#define	_mm_cmple_ps(_v,_w) (__m128i)vec_cmple(_v,_w)
#define	_mm_cmplt_ps(_v,_w) (__m128i)vec_cmplt(_v,_w)
#define	_mm_cmpeq_ps(_v,_w) (__m128i)vec_cmpeq(_v,_w)
#define	_mm_cmp_ps(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))
#define	_mm_cmp_pd(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))
#define	_mm_cmp_ss(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))
#define	_mm_cmp_sd(_v,_w,_c) _EXPAND(_DEFER(_CAT(_,_c))(_v,_w))

/*
 * More macros that have to have secondary expansion.
 */

#define __MM_FROUND_TO_ZERO(_v) vec_trunc(_v)
// - does seem to exist with GCC 5.4 #define __MM_FROUND_TO_ZERO(_v) vec_roundz(_v)
#define	_mm_round_ps(_v,_m) _EXPAND(_DEFER(_CAT(_,_m))(_v))
#define	_mm_round_pd(_v,_m) _EXPAND(_DEFER(_CAT(_,_m))(_v))
#endif


#ifdef	DEBUG
#include <stdio.h>
static inline void
_dumpfvec(__m128 a, char *t)
{
  int i;
  printf("%s:", t);
  for (i = 0 ; i < 4 ; i++) {
    printf(" %#x", *(unsigned int *)&a[i]);
  }
  printf("\n");
}
static inline void
_dumpdvec(__m128d a, char *t)
{
  int i;
  printf("%s:", t);
  for (i = 0 ; i < 2 ; i++) {
    printf(" %#lx", *(unsigned long int *)&a[i]);
  }
  printf("\n");
}

#endif
