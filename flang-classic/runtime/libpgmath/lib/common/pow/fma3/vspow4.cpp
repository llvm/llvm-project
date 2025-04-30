
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if defined(TARGET_LINUX_POWER)
#include "xmm2altivec.h"
#else
#include <immintrin.h>
#endif
#include "pow_defs.h"

extern "C" __m128 __fvs_pow_fma3(__m128 const, __m128 const);

#if defined(TARGET_LINUX_POWER)
/*
 * X86-64 implementation of __fsv_pow_fma3 uses AVX instructions to compute
 * intermediary results in double precision floating point using YMM (256-bit)
 * registers.
 *
 * The POWER architecture is (currently?) limited to 128-bit vector registers.
 *
 * Thus, the POWER implementation will make two calls to the double
 * precision version of __fdv_pow_fma3 to compute the results.
 */

extern "C" __m128d __fvd_pow_fma3(__m128d const a, __m128d const b);

__m128 __fvs_pow_fma3(__m128 const a, __m128 const b)
{
  double da[4] __attribute__((aligned(16)));
  double db[4] __attribute__((aligned(16)));
  double dr[4] __attribute__((aligned(16)));
  float fa[4] __attribute__((aligned(16)));
  float fb[4] __attribute__((aligned(16)));
  float fr[4] __attribute__((aligned(16)));
  __m128d vda;
  __m128d vdb;
  __m128d vdr;
  int i;

  vec_st(a, 0, fa);
  vec_st(b, 0, fb);
  vda = vec_insert((double)fa[0], vda, 0);
  vda = vec_insert((double)fa[1], vda, 1);
  vdb = vec_insert((double)fb[0], vdb, 0);
  vdb = vec_insert((double)fb[1], vdb, 1);
  vdr =  __fvd_pow_fma3(vda, vdb);
  vec_st(vdr, 0, (__m128d *)dr);

  vda = vec_insert((double)fa[2], vda, 0);
  vda = vec_insert((double)fa[3], vda, 1);
  vdb = vec_insert((double)fb[2], vdb, 0);
  vdb = vec_insert((double)fb[3], vdb, 1);
  vdr =  __fvd_pow_fma3(vda, vdb);
  vec_st(vdr, 16, (__m128d *)dr);

  for (i = 0; i < 4 ; i++) {
    fr[i] = dr[i];
  }

  return vec_ld(0, (__m128 *)fr);;
}
#else //defined(TARGET_LINUX_POWER)

__m128 __attribute__ ((noinline)) __pgm_pow_vec128_dp_special_cases(__m128 res_exp, __m128 const a, __m128 const b)
{
   __m128i abs_mask = _mm_set1_epi32(D_ABS_MASK);
   __m128i pos_inf = _mm_set1_epi32(D_POS_INF);
   __m128i sign_mask = _mm_set1_epi32(D_SIGN_MASK);
   __m128i neg_inf = _mm_set1_epi32(D_NEG_INF);
   __m128i nan = _mm_set1_epi32(D_NAN);
   __m128i neg_nan = _mm_set1_epi32(D_NEG_NAN);
   __m128 MINUS_ONE_F_VEC = _mm_set1_ps(D_MINUS_ONE_F);
   __m128 MINUS_ZERO_F_VEC = _mm_set1_ps(D_MINUS_ZERO_F);
   __m128  const ONE_F_VEC = _mm_set1_ps(D_ONE_F);
   __m128  const ZERO_F_VEC = _mm_setzero_ps();


   __m128i b_is_nan = (__m128i)_mm_cmp_ps(b, b, _CMP_NEQ_UQ);
   __m128i a_is_nan = (__m128i)_mm_cmp_ps(a,a, _CMP_NEQ_UQ);
   __m128i a_is_neg = (__m128i)_mm_cmp_ps(a, ZERO_F_VEC, _CMP_LT_OS);
   int a_is_neg_flag = _mm_movemask_epi8((__m128i)a_is_neg);

   __m128i b_is_integer = (__m128i)_mm_cmp_ps(b, _mm_floor_ps(b), _CMP_EQ_OQ);
   int b_is_integer_flag = _mm_movemask_epi8((__m128i)b_is_integer);

   __m128i b_is_odd_integer = _mm_and_si128(b_is_integer, 
                                            _mm_cmpeq_epi32(
                                                            _mm_and_si128(_mm_cvtps_epi32(b), _mm_set1_epi32(0x1)),
                                                            _mm_set1_epi32(0x1)));

   __m128i b_is_even_integer = _mm_and_si128(b_is_integer, 
                                             _mm_cmpeq_epi32(
                                                             _mm_and_si128(_mm_cvtps_epi32(b), _mm_set1_epi32(0x1)),
                                                             _mm_set1_epi32(0x0)));

   __m128i b_is_lt_zero = (__m128i)_mm_cmp_ps(b, ZERO_F_VEC, _CMP_LT_OS);
   int b_is_lt_zero_flag = _mm_movemask_epi8((__m128i)b_is_lt_zero);

   __m128i b_is_gt_zero = (__m128i)_mm_cmp_ps(b, ZERO_F_VEC, _CMP_GT_OS);

   __m128i b_is_odd_integer_lt_zero = _mm_and_si128( b_is_integer,
                                                     (__m128i)_mm_cmp_ps(b, ZERO_F_VEC, _CMP_LT_OS));

   __m128i b_is_odd_integer_gt_zero = _mm_and_si128( b_is_integer,
                                                     (__m128i)_mm_cmp_ps(b, ZERO_F_VEC, _CMP_GT_OS));

   __m128i change_sign_mask = _mm_and_si128(a_is_neg, b_is_odd_integer);
   __m128 changed_sign = _mm_xor_ps( res_exp, (__m128)sign_mask);
   res_exp = _mm_or_ps( _mm_and_ps((__m128)change_sign_mask, (__m128)changed_sign), _mm_andnot_ps((__m128)change_sign_mask, res_exp));

   __m128i return_neg_nan_mask = _mm_andnot_si128(b_is_integer, a_is_neg);
   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_neg_nan_mask, (__m128)neg_nan), _mm_andnot_ps((__m128)return_neg_nan_mask, res_exp));

   __m128i return_nan_mask = _mm_or_si128( b_is_nan, a_is_nan);

   __m128 b_as_nan = _mm_or_ps( _mm_and_ps(
                                           (__m128)b_is_nan, _mm_or_ps(b, (__m128)_mm_set1_epi32(0x00400000))),
                                _mm_andnot_ps((__m128)b_is_nan, res_exp));

   __m128 a_as_nan = _mm_or_ps( _mm_and_ps(
                                           (__m128)a_is_nan, _mm_or_ps(a, (__m128)_mm_set1_epi32(0x00400000))),
                                _mm_andnot_ps((__m128)a_is_nan, res_exp));

   __m128 nan_to_return = _mm_and_ps((__m128)b_is_nan, b_as_nan);
   nan_to_return = _mm_or_ps( _mm_and_ps((__m128)a_is_nan, a_as_nan), _mm_andnot_ps((__m128)a_is_nan,nan_to_return));



   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_nan_mask, (__m128)nan_to_return), _mm_andnot_ps((__m128)return_nan_mask, res_exp));

   __m128i b_is_neg_inf = _mm_cmpeq_epi32( (__m128i)b, neg_inf);
   __m128i b_is_pos_inf = _mm_cmpeq_epi32( (__m128i)b, pos_inf);
   __m128i b_is_any_inf = _mm_or_si128( b_is_pos_inf, b_is_neg_inf);

   __m128i a_is_neg_inf = _mm_cmpeq_epi32( (__m128i)a, neg_inf);
   __m128i a_is_pos_inf = _mm_cmpeq_epi32( (__m128i)a, pos_inf);
   __m128i a_is_any_inf = _mm_or_si128( a_is_pos_inf, a_is_neg_inf);

   __m128i a_is_pos_zero = _mm_cmpeq_epi32( (__m128i)a, (__m128i)ZERO_F_VEC);
   __m128i a_is_neg_zero = _mm_cmpeq_epi32( (__m128i)a, (__m128i)MINUS_ZERO_F_VEC);
   __m128i a_is_any_zero = _mm_or_si128(a_is_pos_zero, a_is_neg_zero);
   int a_is_any_zero_flag = _mm_movemask_epi8((__m128i)a_is_any_zero);

   __m128i abs_a = _mm_and_si128( (__m128i)a, abs_mask);
   __m128 abs_a_lt_one = _mm_cmp_ps( (__m128)abs_a, ONE_F_VEC, _CMP_LT_OS);
   __m128 abs_a_gt_one = _mm_cmp_ps( (__m128)abs_a, ONE_F_VEC, _CMP_GT_OS);

   __m128i a_is_one_mask = _mm_cmpeq_epi32( (__m128i)a, (__m128i)ONE_F_VEC);
   __m128i a_is_minus_one_mask = _mm_cmpeq_epi32( (__m128i)a, (__m128i)MINUS_ONE_F_VEC);

   __m128i return_1_mask = _mm_or_si128( a_is_one_mask, (__m128i)_mm_cmp_ps( b, ZERO_F_VEC, _CMP_EQ_OQ));
   return_1_mask = _mm_or_si128(return_1_mask, _mm_and_si128( a_is_minus_one_mask, b_is_any_inf));
   return_1_mask = _mm_or_si128(return_1_mask, _mm_and_si128( a_is_minus_one_mask, b_is_even_integer));


   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_1_mask, ONE_F_VEC), _mm_andnot_ps((__m128)return_1_mask, res_exp));


   __m128i return_minus_1_mask = _mm_and_si128( a_is_minus_one_mask, b_is_odd_integer );
   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_minus_1_mask, MINUS_ONE_F_VEC), _mm_andnot_ps((__m128)return_minus_1_mask, res_exp));



   __m128i return_neg_zero_mask = _mm_and_si128(a_is_neg_inf,
                                                b_is_odd_integer_lt_zero);
   return_neg_zero_mask = _mm_or_si128(return_neg_zero_mask, _mm_and_si128( a_is_neg_zero, b_is_odd_integer_gt_zero));
   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_neg_zero_mask, MINUS_ZERO_F_VEC), _mm_andnot_ps((__m128)return_neg_zero_mask, res_exp));




   __m128i return_pos_zero_mask = _mm_and_si128( (__m128i)abs_a_gt_one, b_is_neg_inf);
   return_pos_zero_mask = _mm_or_si128(return_pos_zero_mask, _mm_and_si128( (__m128i)abs_a_lt_one, b_is_pos_inf));
   return_pos_zero_mask = _mm_or_si128(return_pos_zero_mask, _mm_and_si128( (__m128i)a_is_neg_inf, _mm_andnot_si128(b_is_odd_integer, b_is_lt_zero)));
   return_pos_zero_mask = _mm_or_si128(return_pos_zero_mask, _mm_and_si128( a_is_pos_zero, b_is_odd_integer_gt_zero));
   return_pos_zero_mask = _mm_or_si128(return_pos_zero_mask, _mm_and_si128( a_is_any_zero, _mm_andnot_si128(b_is_odd_integer, b_is_gt_zero)));
   return_pos_zero_mask = _mm_or_si128(return_pos_zero_mask, _mm_and_si128( a_is_pos_inf, b_is_lt_zero));


   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_pos_zero_mask, ZERO_F_VEC), _mm_andnot_ps((__m128)return_pos_zero_mask, res_exp));

   __m128i return_neg_inf_mask = _mm_and_si128(a_is_neg_inf, _mm_and_si128(b_is_odd_integer, b_is_gt_zero));
   return_neg_inf_mask= _mm_or_si128(return_neg_inf_mask, _mm_and_si128(a_is_neg_zero, b_is_odd_integer_lt_zero));

   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_neg_inf_mask, (__m128)neg_inf), _mm_andnot_ps((__m128)return_neg_inf_mask, res_exp));


   __m128i return_pos_inf_mask = _mm_and_si128( (__m128i)abs_a_lt_one, b_is_neg_inf);
   return_pos_inf_mask= _mm_or_si128(return_pos_inf_mask, _mm_and_si128( (__m128i)abs_a_gt_one, b_is_pos_inf));
   return_pos_inf_mask= _mm_or_si128(return_pos_inf_mask, _mm_and_si128(a_is_pos_zero, b_is_odd_integer_lt_zero));
   return_pos_inf_mask= _mm_or_si128(return_pos_inf_mask, _mm_and_si128(a_is_neg_inf, _mm_andnot_si128(b_is_odd_integer, b_is_gt_zero)));
   return_pos_inf_mask= _mm_or_si128(return_pos_inf_mask, _mm_and_si128(a_is_any_zero, _mm_andnot_si128(b_is_odd_integer, b_is_lt_zero)));
   return_pos_inf_mask= _mm_or_si128(return_pos_inf_mask, _mm_and_si128(a_is_pos_inf, b_is_gt_zero));

   res_exp = _mm_or_ps( _mm_and_ps((__m128)return_pos_inf_mask, (__m128)pos_inf), _mm_andnot_ps((__m128)return_pos_inf_mask, res_exp));

/*
 *  Before returning see if we need to set any of the processor
 *  exception flags.
 *
 *  Domain error:  a is negative, and b is a finite noninteger
 *  we need to raise the Invalid-Operation flag.   This can be done by
 *  taking the square root of a negative number.
 * 
 *  Pole error:  a is zero and b is negative we need to raise the
 *  divide by zero flag.   This can be done by dividing by zero.
 */

   if (a_is_neg_flag && (!b_is_integer_flag)) {
      __m128 volatile invop = _mm_sqrt_ps(a);
   }

   if (a_is_any_zero_flag && b_is_lt_zero_flag) {
      __m128 volatile divXzero = _mm_div_ps(ONE_F_VEC,ZERO_F_VEC);
   }

   return res_exp;
}

__m128 __fvs_pow_fma3(__m128 const a, __m128 const b)
{
   //   fpminimax(log2(x),10,[|double...|],[0.5;0.9999999],relative);
   __m256d const LOG_C0_VEC   = _mm256_set1_pd(LOG_C0);
   __m256d const LOG_C1_VEC   = _mm256_set1_pd(LOG_C1);
   __m256d const LOG_C2_VEC   = _mm256_set1_pd(LOG_C2);
   __m256d const LOG_C3_VEC   = _mm256_set1_pd(LOG_C3);
   __m256d const LOG_C4_VEC   = _mm256_set1_pd(LOG_C4);
   __m256d const LOG_C5_VEC   = _mm256_set1_pd(LOG_C5);
   __m256d const LOG_C6_VEC   = _mm256_set1_pd(LOG_C6);
   __m256d const LOG_C7_VEC   = _mm256_set1_pd(LOG_C7);
   __m256d const LOG_C8_VEC   = _mm256_set1_pd(LOG_C8);
   __m256d const LOG_C9_VEC   = _mm256_set1_pd(LOG_C9);
   __m256d const LOG_C10_VEC  = _mm256_set1_pd(LOG_C10);

   //   fpminimax(exp(x*0.6931471805599453094172321214581765680755001343602552),6,[|double...|],[-0.5,0.5],relative);
   __m256d const EXP_C0_VEC = _mm256_set1_pd(EXP_C0);
   __m256d const EXP_C1_VEC = _mm256_set1_pd(EXP_C1);
   __m256d const EXP_C2_VEC = _mm256_set1_pd(EXP_C2);
   __m256d const EXP_C3_VEC = _mm256_set1_pd(EXP_C3);
   __m256d const EXP_C4_VEC = _mm256_set1_pd(EXP_C4);
   __m256d const EXP_C5_VEC = _mm256_set1_pd(EXP_C5);
   __m256d const EXP_C6_VEC = _mm256_set1_pd(EXP_C6);

   __m128  const ONE_F_VEC = _mm_set1_ps(D_ONE_F);
   __m128  const ZERO_F_VEC = _mm_setzero_ps();
   __m128i const ALL_ONES_EXPONENT = _mm_set1_epi32(D_ALL_ONES_EXPONENT);

   __m128i const bit_mask2 = _mm_set1_epi32(D_BIT_MASK2);
   __m128i exp_offset = _mm_set1_epi32(D_EXP_OFFSET);
   __m128i const offset = _mm_set1_epi32(D_OFFSET);

   __m256d const EXP_HI_VEC = _mm256_set1_pd(EXP_HI);
   __m256d const EXP_LO_VEC = _mm256_set1_pd(EXP_LO);
   __m256d const DBL2INT_CVT_VEC= _mm256_set1_pd(DBL2INT_CVT);

   __m128 const TWO_TO_M126_F_VEC = _mm_set1_ps(0x1p-126f);
   __m128i const U24_VEC = _mm_set1_epi32(D_U24);
   __m128 const TWO_TO_24_F_VEC = _mm_set1_ps(D_TWO_TO_24_F);
   __m128i sign_mask2 = _mm_set1_epi32(D_SIGN_MASK2);

   __m128 a_compute = _mm_and_ps(a, (__m128)sign_mask2);

   __m128 res;

   __m256d b_d = _mm256_cvtps_pd(b);

   __m128 mask = (__m128)_mm_cmp_ps((__m128)a_compute, TWO_TO_M126_F_VEC, _CMP_LT_OS);
   int moved_mask = _mm_movemask_ps(mask);
   if (moved_mask) {
      a_compute= _mm_or_ps( _mm_and_ps(mask, _mm_mul_ps(a_compute, TWO_TO_24_F_VEC)), _mm_andnot_ps(mask,a_compute));
      exp_offset = _mm_add_epi32(exp_offset, _mm_and_si128((__m128i)mask, U24_VEC));
   }

   __m128i e_int = _mm_sub_epi32(_mm_srli_epi32( (__m128i)a_compute, 23), exp_offset);

   __m256d e = _mm256_cvtepi32_pd(e_int);

   __m128 detect_inf_nan = _mm_add_ps(a_compute, b);
   __m128i overridemask = _mm_cmpeq_epi32( (__m128i)a_compute, (__m128i)ONE_F_VEC);
   overridemask = _mm_or_si128( overridemask, (__m128i)_mm_cmp_ps( b, ZERO_F_VEC, _CMP_EQ_OQ));
   overridemask = _mm_or_si128( overridemask, _mm_cmpeq_epi32( _mm_and_si128((__m128i)detect_inf_nan, ALL_ONES_EXPONENT), (__m128i)ALL_ONES_EXPONENT));
   overridemask = _mm_or_si128( overridemask, (__m128i)_mm_cmp_ps(a, ZERO_F_VEC, _CMP_LE_OQ));
   int reducedMask = _mm_movemask_epi8(overridemask);

   __m128 m = (__m128)_mm_add_epi32(_mm_and_si128( (__m128i)a_compute, bit_mask2), offset);
   __m256d m_d = _mm256_cvtps_pd(m);


   //   __m256d t_hi = LOG_C0_VEC;
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C1_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C2_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C3_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C4_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C5_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C6_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C7_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C8_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C9_VEC);
   //   t_hi = _mm256_fmadd_pd(t_hi, m_hi, LOG_C10_VEC);

   __m256d m2 = _mm256_mul_pd(m_d, m_d);
   __m256d m4 = _mm256_mul_pd(m2, m2);


   __m256d a1 = _mm256_fmadd_pd(m_d, LOG_C9_VEC, LOG_C10_VEC);
   __m256d a2 = _mm256_fmadd_pd(m_d, LOG_C7_VEC, LOG_C8_VEC);
   __m256d a3 = _mm256_fmadd_pd(m_d, LOG_C5_VEC, LOG_C6_VEC);
   __m256d a4 = _mm256_fmadd_pd(m_d, LOG_C3_VEC, LOG_C4_VEC);
   __m256d a5 = _mm256_fmadd_pd(m_d, LOG_C1_VEC, LOG_C2_VEC);
   __m256d a6 = _mm256_mul_pd(LOG_C0_VEC, m2);


   __m256d a7 = _mm256_fmadd_pd(m2, a2, a1);
   __m256d a8 = _mm256_fmadd_pd(m2, a4, a3);
   __m256d a9 = _mm256_add_pd(a5, a6);

   __m256d a10 = _mm256_fmadd_pd(m4, a9, a8);
   __m256d t = _mm256_fmadd_pd(m4, a10, a7);

   t = _mm256_add_pd(e, t);
   __m256d temp = _mm256_mul_pd(b_d, t);

   //---------exponent starts here
   //   __m256i exp_override = (__m256i)_mm256_cmp_pd( temp_hi, EXP_HI_VEC, _CMP_GT_OS);
   //   exp_override = _mm256_or_si256(exp_override, (__m256i)_mm256_cmp_pd(temp_lo, EXP_HI_VEC, _CMP_GT_OS));
   //   exp_override = _mm256_or_si256(exp_override, (__m256i)_mm256_cmp_pd(temp_hi, EXP_LO_VEC, _CMP_LT_OS));
   //   exp_override = _mm256_or_si256(exp_override, (__m256i)_mm256_cmp_pd(temp_lo, EXP_LO_VEC, _CMP_LT_OS));
   //   int exp_reduced_mask= _mm256_movemask_epi8(exp_override);
   //   if (exp_reduced_mask) {
   //      return pow_vec256_dp_slowpath(a, b);
   //   }

   temp = _mm256_min_pd(temp,EXP_HI_VEC );
   temp = _mm256_max_pd(temp,EXP_LO_VEC );

   __m256d t_exp = _mm256_add_pd(temp, DBL2INT_CVT_VEC);

   __m256d tt = _mm256_sub_pd(t_exp, DBL2INT_CVT_VEC);
   __m256i integer = _mm256_castpd_si256(t_exp);
   __m256d z_exp = _mm256_sub_pd( temp, tt);

   __m256d poly_exp;
   poly_exp = EXP_C0_VEC;
   poly_exp = _mm256_fmadd_pd(poly_exp, z_exp, EXP_C1_VEC);
   poly_exp = _mm256_fmadd_pd(poly_exp, z_exp, EXP_C2_VEC);
   poly_exp = _mm256_fmadd_pd(poly_exp, z_exp, EXP_C3_VEC);
   poly_exp = _mm256_fmadd_pd(poly_exp, z_exp, EXP_C4_VEC);
   poly_exp = _mm256_fmadd_pd(poly_exp, z_exp, EXP_C5_VEC);
   poly_exp = _mm256_fmadd_pd(poly_exp, z_exp, EXP_C6_VEC);

   __m256i integer_poly_exp = _mm256_castpd_si256(poly_exp);
   integer = _mm256_slli_epi64(integer, 52);
   integer_poly_exp = _mm256_add_epi32(integer, integer_poly_exp);
   __m128 res_exp = _mm256_cvtpd_ps((__m256d)integer_poly_exp);

   if( __builtin_expect(reducedMask,0)) {
      return __pgm_pow_vec128_dp_special_cases(res_exp, a, b);
   }
   return res_exp;
}
#endif //defined(TARGET_LINUX_POWER)
