
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if defined(TARGET_LINUX_POWER)
#error "Source cannot be compiled for POWER architectures"
#include "xmm2altivec.h"
#else
#include <immintrin.h>
#include "mth_avx512helper.h"
#endif
#include "pow_defs.h"

extern "C" __m512 FCN_AVX512(__fvs_pow_fma3)(__m512 const, __m512 const);


static __m512 __attribute__ ((noinline)) __pgm_pow_vec512_dp_special_cases(__m512 res_exp, __m512 const a, __m512 const b)
{
      __m512i abs_mask = _mm512_set1_epi32(D_SIGN_MASK2);
      __m512i pos_inf = _mm512_set1_epi32(D_POS_INF);
      __m512i sign_mask = _mm512_set1_epi32(D_SIGN_MASK);
      __m512i neg_inf = _mm512_set1_epi32(D_NEG_INF);
      __m512i nan = _mm512_set1_epi32(D_NAN);
      __m512i neg_nan = _mm512_set1_epi32(D_NEG_NAN);
      __m512 MINUS_ONE_F_VEC = _mm512_set1_ps(D_MINUS_ONE_F);
      __m512 MINUS_ZERO_F_VEC = _mm512_set1_ps(D_MINUS_ZERO_F);
      __m512  const ONE_F_VEC = _mm512_set1_ps(D_ONE_F);
      __m512  const ZERO_F_VEC = _mm512_setzero_ps();


      __m512i b_is_nan = (__m512i)_MM512_CMP_PS(b, b, _CMP_NEQ_UQ);
      __m512i a_is_nan = (__m512i)_MM512_CMP_PS(a,a, _CMP_NEQ_UQ);
      __m512i a_is_neg = (__m512i)_MM512_CMP_PS(a, ZERO_F_VEC, _CMP_LT_OS);
      int a_is_neg_flag = _MM512_MOVEMASK_PS(_mm512_castsi512_ps(a_is_neg));

      __m512i b_is_integer = (__m512i)_MM512_CMP_PS(b, _mm512_floor_ps(b), _CMP_EQ_OQ);
      int b_is_integer_flag = _MM512_MOVEMASK_PS(_mm512_castsi512_ps(b_is_integer));

      __m512i b_is_odd_integer = _mm512_and_si512(b_is_integer, 
                                                  _MM512_CMPEQ_EPI32(
                                                                     _mm512_and_si512(_mm512_cvtps_epi32(b), _mm512_set1_epi32(0x1)),
                                                                     _mm512_set1_epi32(0x1)));

      __m512i b_is_even_integer = _mm512_and_si512(b_is_integer, 
                                                  _MM512_CMPEQ_EPI32(
                                                                     _mm512_and_si512(_mm512_cvtps_epi32(b), _mm512_set1_epi32(0x1)),
                                                                     _mm512_set1_epi32(0x0)));

      __m512i b_is_lt_zero = (__m512i)_MM512_CMP_PS(b, ZERO_F_VEC, _CMP_LT_OS);
      int b_is_lt_zero_flag = _MM512_MOVEMASK_PS(_mm512_castsi512_ps(b_is_lt_zero));

      __m512i b_is_gt_zero = (__m512i)_MM512_CMP_PS(b, ZERO_F_VEC, _CMP_GT_OS);

      __m512i b_is_odd_integer_lt_zero = _mm512_and_si512( b_is_integer,
                                                           (__m512i)_MM512_CMP_PS(b, ZERO_F_VEC, _CMP_LT_OS));

      __m512i b_is_odd_integer_gt_zero = _mm512_and_si512( b_is_integer,
                                                           (__m512i)_MM512_CMP_PS(b, ZERO_F_VEC, _CMP_GT_OS));

      __m512i change_sign_mask = _mm512_and_si512(a_is_neg, b_is_odd_integer);
      __m512 changed_sign = _MM512_XOR_PS( res_exp, (__m512)sign_mask);
      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)change_sign_mask, (__m512)changed_sign), _MM512_ANDNOT_PS((__m512)change_sign_mask, res_exp));

      __m512i return_neg_nan_mask = _mm512_andnot_si512(b_is_integer, a_is_neg);
      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_neg_nan_mask, (__m512)neg_nan), _MM512_ANDNOT_PS((__m512)return_neg_nan_mask, res_exp));

      __m512i return_nan_mask = _mm512_or_si512( b_is_nan, a_is_nan);

      __m512 b_as_nan = _MM512_OR_PS( _MM512_AND_PS(
                                                     (__m512)b_is_nan, _MM512_OR_PS(b, (__m512)_mm512_set1_epi32(0x00400000))),
                                                     _MM512_ANDNOT_PS((__m512)b_is_nan, res_exp));

      __m512 a_as_nan = _MM512_OR_PS( _MM512_AND_PS(
                                                     (__m512)a_is_nan, _MM512_OR_PS(a, (__m512)_mm512_set1_epi32(0x00400000))),
                                                     _MM512_ANDNOT_PS((__m512)a_is_nan, res_exp));

      __m512 nan_to_return = _MM512_AND_PS((__m512)b_is_nan, b_as_nan);
      nan_to_return = _MM512_OR_PS( _MM512_AND_PS((__m512)a_is_nan, a_as_nan), _MM512_ANDNOT_PS((__m512)a_is_nan,nan_to_return));



      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_nan_mask, (__m512)nan_to_return), _MM512_ANDNOT_PS((__m512)return_nan_mask, res_exp));

      __m512i b_is_neg_inf = _MM512_CMPEQ_EPI32( (__m512i)b, neg_inf);
      __m512i b_is_pos_inf = _MM512_CMPEQ_EPI32( (__m512i)b, pos_inf);
      __m512i b_is_any_inf = _mm512_or_si512( b_is_pos_inf, b_is_neg_inf);

      __m512i a_is_neg_inf = _MM512_CMPEQ_EPI32( (__m512i)a, neg_inf);
      __m512i a_is_pos_inf = _MM512_CMPEQ_EPI32( (__m512i)a, pos_inf);
      __m512i a_is_any_inf = _mm512_or_si512( a_is_pos_inf, a_is_neg_inf);

      __m512i a_is_pos_zero = _MM512_CMPEQ_EPI32( (__m512i)a, (__m512i)ZERO_F_VEC);
      __m512i a_is_neg_zero = _MM512_CMPEQ_EPI32( (__m512i)a, (__m512i)MINUS_ZERO_F_VEC);
      __m512i a_is_any_zero = _mm512_or_si512(a_is_pos_zero, a_is_neg_zero);
      int a_is_any_zero_flag = _MM512_MOVEMASK_PS(_mm512_castsi512_ps(a_is_any_zero));

      __m512i abs_a = _mm512_and_si512( (__m512i)a, abs_mask);
      __m512 abs_a_lt_one = _MM512_CMP_PS( (__m512)abs_a, ONE_F_VEC, _CMP_LT_OS);
      __m512 abs_a_gt_one = _MM512_CMP_PS( (__m512)abs_a, ONE_F_VEC, _CMP_GT_OS);

      __m512i a_is_one_mask = _MM512_CMPEQ_EPI32( (__m512i)a, (__m512i)ONE_F_VEC);
      __m512i a_is_minus_one_mask = _MM512_CMPEQ_EPI32( (__m512i)a, (__m512i)MINUS_ONE_F_VEC);

      __m512i return_1_mask = _mm512_or_si512( a_is_one_mask, (__m512i)_MM512_CMP_PS( b, ZERO_F_VEC, _CMP_EQ_OQ));
      return_1_mask = _mm512_or_si512(return_1_mask, _mm512_and_si512( a_is_minus_one_mask, b_is_any_inf));
      return_1_mask = _mm512_or_si512(return_1_mask, _mm512_and_si512( a_is_minus_one_mask, b_is_even_integer));


      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_1_mask, ONE_F_VEC), _MM512_ANDNOT_PS((__m512)return_1_mask, res_exp));


      __m512i return_minus_1_mask = _mm512_and_si512( a_is_minus_one_mask, b_is_odd_integer );
      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_minus_1_mask, MINUS_ONE_F_VEC), _MM512_ANDNOT_PS((__m512)return_minus_1_mask, res_exp));



      __m512i return_neg_zero_mask = _mm512_and_si512(a_is_neg_inf,
                                                      b_is_odd_integer_lt_zero);
      return_neg_zero_mask = _mm512_or_si512(return_neg_zero_mask, _mm512_and_si512( a_is_neg_zero, b_is_odd_integer_gt_zero));
      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_neg_zero_mask, MINUS_ZERO_F_VEC), _MM512_ANDNOT_PS((__m512)return_neg_zero_mask, res_exp));




      __m512i return_pos_zero_mask = _mm512_and_si512( (__m512i)abs_a_gt_one, b_is_neg_inf);
      return_pos_zero_mask = _mm512_or_si512(return_pos_zero_mask, _mm512_and_si512( (__m512i)abs_a_lt_one, b_is_pos_inf));
      return_pos_zero_mask = _mm512_or_si512(return_pos_zero_mask, _mm512_and_si512( (__m512i)a_is_neg_inf, _mm512_andnot_si512(b_is_odd_integer, b_is_lt_zero)));
      return_pos_zero_mask = _mm512_or_si512(return_pos_zero_mask, _mm512_and_si512( a_is_pos_zero, b_is_odd_integer_gt_zero));
      return_pos_zero_mask = _mm512_or_si512(return_pos_zero_mask, _mm512_and_si512( a_is_any_zero, _mm512_andnot_si512(b_is_odd_integer, b_is_gt_zero)));
      return_pos_zero_mask = _mm512_or_si512(return_pos_zero_mask, _mm512_and_si512( a_is_pos_inf, b_is_lt_zero));


      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_pos_zero_mask, ZERO_F_VEC), _MM512_ANDNOT_PS((__m512)return_pos_zero_mask, res_exp));

      __m512i return_neg_inf_mask = _mm512_and_si512(a_is_neg_inf, _mm512_and_si512(b_is_odd_integer, b_is_gt_zero));
      return_neg_inf_mask= _mm512_or_si512(return_neg_inf_mask, _mm512_and_si512(a_is_neg_zero, b_is_odd_integer_lt_zero));

      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_neg_inf_mask, (__m512)neg_inf), _MM512_ANDNOT_PS((__m512)return_neg_inf_mask, res_exp));


      __m512i return_pos_inf_mask = _mm512_and_si512( (__m512i)abs_a_lt_one, b_is_neg_inf);
      return_pos_inf_mask= _mm512_or_si512(return_pos_inf_mask, _mm512_and_si512( (__m512i)abs_a_gt_one, b_is_pos_inf));
      return_pos_inf_mask= _mm512_or_si512(return_pos_inf_mask, _mm512_and_si512(a_is_pos_zero, b_is_odd_integer_lt_zero));
      return_pos_inf_mask= _mm512_or_si512(return_pos_inf_mask, _mm512_and_si512(a_is_neg_inf, _mm512_andnot_si512(b_is_odd_integer, b_is_gt_zero)));
      return_pos_inf_mask= _mm512_or_si512(return_pos_inf_mask, _mm512_and_si512(a_is_any_zero, _mm512_andnot_si512(b_is_odd_integer, b_is_lt_zero)));
      return_pos_inf_mask= _mm512_or_si512(return_pos_inf_mask, _mm512_and_si512(a_is_pos_inf, b_is_gt_zero));

      res_exp = _MM512_OR_PS( _MM512_AND_PS((__m512)return_pos_inf_mask, (__m512)pos_inf), _MM512_ANDNOT_PS((__m512)return_pos_inf_mask, res_exp));

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
         __m512 volatile invop = _mm512_sqrt_ps(a);
      }

      if (a_is_any_zero_flag && b_is_lt_zero_flag) {
         __m512 volatile divXzero = _mm512_div_ps(ONE_F_VEC,ZERO_F_VEC);
      }

      return res_exp;
}

__m512 FCN_AVX512(__fvs_pow_fma3)(__m512 const a, __m512 const b)
{
//   fpminimax(log2(x),10,[|double...|],[0.5;0.9999999],relative);
   __m512d const LOG_C0_VEC   = _mm512_set1_pd(LOG_C0);
   __m512d const LOG_C1_VEC   = _mm512_set1_pd(LOG_C1);
   __m512d const LOG_C2_VEC   = _mm512_set1_pd(LOG_C2);
   __m512d const LOG_C3_VEC   = _mm512_set1_pd(LOG_C3);
   __m512d const LOG_C4_VEC   = _mm512_set1_pd(LOG_C4);
   __m512d const LOG_C5_VEC   = _mm512_set1_pd(LOG_C5);
   __m512d const LOG_C6_VEC   = _mm512_set1_pd(LOG_C6);
   __m512d const LOG_C7_VEC   = _mm512_set1_pd(LOG_C7);
   __m512d const LOG_C8_VEC   = _mm512_set1_pd(LOG_C8);
   __m512d const LOG_C9_VEC   = _mm512_set1_pd(LOG_C9);
   __m512d const LOG_C10_VEC  = _mm512_set1_pd(LOG_C10);

//   fpminimax(exp(x*0.6931471805599453094172321214581765680755001343602552),6,[|double...|],[-0.5,0.5],relative);
   __m512d const EXP_C0_VEC = _mm512_set1_pd(EXP_C0);
   __m512d const EXP_C1_VEC = _mm512_set1_pd(EXP_C1);
   __m512d const EXP_C2_VEC = _mm512_set1_pd(EXP_C2);
   __m512d const EXP_C3_VEC = _mm512_set1_pd(EXP_C3);
   __m512d const EXP_C4_VEC = _mm512_set1_pd(EXP_C4);
   __m512d const EXP_C5_VEC = _mm512_set1_pd(EXP_C5);
   __m512d const EXP_C6_VEC = _mm512_set1_pd(EXP_C6);

   __m512  const ONE_F_VEC = _mm512_set1_ps(D_ONE_F);
   __m512  const ZERO_F_VEC = _mm512_setzero_ps();
   __m512i const ALL_ONES_EXPONENT = _mm512_set1_epi32(D_ALL_ONES_EXPONENT);

   __m512i const bit_mask2 = _mm512_set1_epi32(D_BIT_MASK2);
   __m512i exp_offset = _mm512_set1_epi32(D_EXP_OFFSET);
   __m512i const offset = _mm512_set1_epi32(D_OFFSET);

   __m512d const EXP_HI_VEC = _mm512_set1_pd(EXP_HI);
   __m512d const EXP_LO_VEC = _mm512_set1_pd(EXP_LO);
   __m512d const DBL2INT_CVT_VEC= _mm512_set1_pd(DBL2INT_CVT);

   __m512 const TWO_TO_M126_F_VEC = _mm512_set1_ps(0x1p-126f);
   __m512i const U24_VEC = _mm512_set1_epi32(D_U24);
   __m512 const TWO_TO_24_F_VEC = _mm512_set1_ps(D_TWO_TO_24_F);
   __m512i sign_mask2 = _mm512_set1_epi32(D_SIGN_MASK2);

   __m512 a_compute = _MM512_AND_PS(a, (__m512)sign_mask2);

   __m512 res;
   __m256 b_hi = _MM512_EXTRACTF256_PS(b, 1);
   __m256 b_lo = _MM512_EXTRACTF256_PS(b, 0);

   __m512d b_hi_d = _mm512_cvtps_pd(b_hi);
   __m512d b_lo_d = _mm512_cvtps_pd(b_lo);

   __m512 mask = (__m512)_MM512_CMP_PS((__m512)a_compute, TWO_TO_M126_F_VEC, _CMP_LT_OS);
   int moved_mask = _MM512_MOVEMASK_PS(mask);
   if (moved_mask) {
      a_compute= _MM512_OR_PS( _MM512_AND_PS(mask, _mm512_mul_ps(a_compute, TWO_TO_24_F_VEC)), _MM512_ANDNOT_PS(mask,a_compute));
      exp_offset = _mm512_add_epi32(exp_offset, _mm512_and_si512((__m512i)mask, U24_VEC));
   }

   __m512i e_int = _mm512_sub_epi32(_mm512_srli_epi32( (__m512i)a_compute, 23), exp_offset);

   __m256i e_int_hi = _MM512_EXTRACTI256_SI512(e_int, 1);
   __m256i e_int_lo = _MM512_EXTRACTI256_SI512(e_int, 0);

   __m512d e_hi = _mm512_cvtepi32_pd(e_int_hi);
   __m512d e_lo = _mm512_cvtepi32_pd(e_int_lo);


   __m512 detect_inf_nan = _mm512_add_ps(a_compute, b);
   __m512i overridemask = _MM512_CMPEQ_EPI32( (__m512i)a_compute, (__m512i)ONE_F_VEC);
   overridemask = _mm512_or_si512( overridemask, (__m512i)_MM512_CMP_PS( b, ZERO_F_VEC, _CMP_EQ_OQ));
   overridemask = _mm512_or_si512( overridemask, _MM512_CMPEQ_EPI32( _mm512_and_si512((__m512i)detect_inf_nan, ALL_ONES_EXPONENT), (__m512i)ALL_ONES_EXPONENT));
   overridemask = _mm512_or_si512( overridemask, (__m512i)_MM512_CMP_PS(a, ZERO_F_VEC, _CMP_LE_OQ));
   int reducedMask = _MM512_MOVEMASK_PS(_mm512_castsi512_ps(overridemask));

   __m512 m = (__m512)_mm512_add_epi32(_mm512_and_si512( (__m512i)a_compute, bit_mask2), offset);
   __m256 m_hi_f = _MM512_EXTRACTF256_PS(m, 1);
   __m256 m_lo_f = _MM512_EXTRACTF256_PS(m, 0);
   __m512d m_hi = _mm512_cvtps_pd(m_hi_f);
   __m512d m_lo = _mm512_cvtps_pd(m_lo_f);

//   __m512d t_hi = LOG_C0_VEC;
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C1_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C2_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C3_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C4_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C5_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C6_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C7_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C8_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C9_VEC);
//   t_hi = _mm512_fmadd_pd(t_hi, m_hi, LOG_C10_VEC);

//   __m512d t_lo = LOG_C0_VEC;
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C1_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C2_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C3_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C4_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C5_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C6_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C7_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C8_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C9_VEC);
//   t_lo = _mm512_fmadd_pd(t_lo, m_lo, LOG_C10_VEC);


   __m512d m2_hi = _mm512_mul_pd(m_hi, m_hi);
   __m512d m4_hi = _mm512_mul_pd(m2_hi, m2_hi);


   __m512d a1_hi = _mm512_fmadd_pd(m_hi, LOG_C9_VEC, LOG_C10_VEC);
   __m512d a2_hi = _mm512_fmadd_pd(m_hi, LOG_C7_VEC, LOG_C8_VEC);
   __m512d a3_hi = _mm512_fmadd_pd(m_hi, LOG_C5_VEC, LOG_C6_VEC);
   __m512d a4_hi = _mm512_fmadd_pd(m_hi, LOG_C3_VEC, LOG_C4_VEC);
   __m512d a5_hi = _mm512_fmadd_pd(m_hi, LOG_C1_VEC, LOG_C2_VEC);
   __m512d a6_hi = _mm512_mul_pd(LOG_C0_VEC, m2_hi);


   __m512d a7_hi = _mm512_fmadd_pd(m2_hi, a2_hi, a1_hi);
   __m512d a8_hi = _mm512_fmadd_pd(m2_hi, a4_hi, a3_hi);
   __m512d a9_hi = _mm512_add_pd(a5_hi, a6_hi);

   __m512d a10_hi = _mm512_fmadd_pd(m4_hi, a9_hi, a8_hi);
   __m512d t_hi = _mm512_fmadd_pd(m4_hi, a10_hi, a7_hi);

   __m512d m2_lo = _mm512_mul_pd(m_lo, m_lo);
   __m512d m4_lo = _mm512_mul_pd(m2_lo, m2_lo);

   __m512d a6_lo = _mm512_mul_pd(LOG_C0_VEC, m2_lo);
   __m512d a1_lo = _mm512_fmadd_pd(m_lo, LOG_C9_VEC, LOG_C10_VEC);
   __m512d a2_lo = _mm512_fmadd_pd(m_lo, LOG_C7_VEC, LOG_C8_VEC);
   __m512d a3_lo = _mm512_fmadd_pd(m_lo, LOG_C5_VEC, LOG_C6_VEC);
   __m512d a4_lo = _mm512_fmadd_pd(m_lo, LOG_C3_VEC, LOG_C4_VEC);
   __m512d a5_lo = _mm512_fmadd_pd(m_lo, LOG_C1_VEC, LOG_C2_VEC);

   __m512d a7_lo = _mm512_fmadd_pd(m2_lo, a2_lo, a1_lo);
   __m512d a8_lo = _mm512_fmadd_pd(m2_lo, a4_lo, a3_lo);
   __m512d a9_lo = _mm512_add_pd(a5_lo, a6_lo);

   __m512d a10_lo = _mm512_fmadd_pd(m4_lo, a9_lo, a8_lo);

   __m512d t_lo = _mm512_fmadd_pd(m4_lo, a10_lo, a7_lo);

   t_lo = _mm512_add_pd(e_lo, t_lo);
   t_hi = _mm512_add_pd(e_hi, t_hi);

   __m512d temp_hi = _mm512_mul_pd(b_hi_d, t_hi);
   __m512d temp_lo = _mm512_mul_pd(b_lo_d, t_lo);

   //---------exponent starts here
//   __m512i exp_override = (__m512i)_mm512_cmp_pd( temp_hi, EXP_HI_VEC, _CMP_GT_OS);
//   exp_override = _mm512_or_si512(exp_override, (__m512i)_mm512_cmp_pd(temp_lo, EXP_HI_VEC, _CMP_GT_OS));
//   exp_override = _mm512_or_si512(exp_override, (__m512i)_mm512_cmp_pd(temp_hi, EXP_LO_VEC, _CMP_LT_OS));
//   exp_override = _mm512_or_si512(exp_override, (__m512i)_mm512_cmp_pd(temp_lo, EXP_LO_VEC, _CMP_LT_OS));
//   int exp_reduced_mask= _mm512_movemask_epi8(exp_override);
//   if (exp_reduced_mask) {
//      return pow_vec512_dp_slowpath(a, b);
//   }

   temp_hi = _mm512_min_pd(temp_hi,EXP_HI_VEC );
   temp_hi = _mm512_max_pd(temp_hi,EXP_LO_VEC );
   temp_lo = _mm512_min_pd(temp_lo,EXP_HI_VEC );
   temp_lo = _mm512_max_pd(temp_lo,EXP_LO_VEC );

   __m512d t_exp_hi = _mm512_add_pd(temp_hi, DBL2INT_CVT_VEC);
   __m512d t_exp_lo = _mm512_add_pd(temp_lo, DBL2INT_CVT_VEC);

   __m512d tt_hi = _mm512_sub_pd(t_exp_hi, DBL2INT_CVT_VEC);
   __m512i integer_hi = _mm512_castpd_si512(t_exp_hi);
   __m512d tt_lo = _mm512_sub_pd(t_exp_lo, DBL2INT_CVT_VEC);
   __m512i integer_lo = _mm512_castpd_si512(t_exp_lo);

   __m512d z_exp_hi = _mm512_sub_pd( temp_hi, tt_hi);
   __m512d z_exp_lo = _mm512_sub_pd( temp_lo, tt_lo);

   __m512d poly_exp_hi;
   poly_exp_hi = EXP_C0_VEC;
   poly_exp_hi = _mm512_fmadd_pd(poly_exp_hi, z_exp_hi, EXP_C1_VEC);
   poly_exp_hi = _mm512_fmadd_pd(poly_exp_hi, z_exp_hi, EXP_C2_VEC);
   poly_exp_hi = _mm512_fmadd_pd(poly_exp_hi, z_exp_hi, EXP_C3_VEC);
   poly_exp_hi = _mm512_fmadd_pd(poly_exp_hi, z_exp_hi, EXP_C4_VEC);
   poly_exp_hi = _mm512_fmadd_pd(poly_exp_hi, z_exp_hi, EXP_C5_VEC);
   poly_exp_hi = _mm512_fmadd_pd(poly_exp_hi, z_exp_hi, EXP_C6_VEC);

   __m512d poly_exp_lo;
   poly_exp_lo = EXP_C0_VEC;
   poly_exp_lo = _mm512_fmadd_pd(poly_exp_lo, z_exp_lo, EXP_C1_VEC);
   poly_exp_lo = _mm512_fmadd_pd(poly_exp_lo, z_exp_lo, EXP_C2_VEC);
   poly_exp_lo = _mm512_fmadd_pd(poly_exp_lo, z_exp_lo, EXP_C3_VEC);
   poly_exp_lo = _mm512_fmadd_pd(poly_exp_lo, z_exp_lo, EXP_C4_VEC);
   poly_exp_lo = _mm512_fmadd_pd(poly_exp_lo, z_exp_lo, EXP_C5_VEC);
   poly_exp_lo = _mm512_fmadd_pd(poly_exp_lo, z_exp_lo, EXP_C6_VEC);

   __m512i integer_poly_exp_hi = _mm512_castpd_si512(poly_exp_hi);
   __m512i integer_poly_exp_lo = _mm512_castpd_si512(poly_exp_lo);
   integer_hi = _mm512_slli_epi64(integer_hi, 52);
   integer_lo = _mm512_slli_epi64(integer_lo, 52);
   integer_poly_exp_hi = _mm512_add_epi32(integer_hi, integer_poly_exp_hi);
   integer_poly_exp_lo = _mm512_add_epi32(integer_lo, integer_poly_exp_lo);

   __m256 res_hi_f = _mm512_cvtpd_ps((__m512d)integer_poly_exp_hi);
   __m256 res_lo_f = _mm512_cvtpd_ps((__m512d)integer_poly_exp_lo);
   __m512 res_exp;
   res_exp = _mm512_castps256_ps512(res_lo_f);
   res_exp = _MM512_INSERTF256_PS(res_exp,res_hi_f,1);

   if( __builtin_expect(reducedMask,0)) {
      return __pgm_pow_vec512_dp_special_cases(res_exp, a, b);
   }
   return res_exp;
}
