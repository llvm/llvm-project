
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if defined(TARGET_LINUX_POWER)
#include "xmm2altivec.h"
#elif defined(TARGET_ARM64)
#include "arm64intrin.h"
#else
#include <immintrin.h>
#endif
#include "pow_defs.h"

extern "C" float __fss_pow_fma3(float const,float const);


float __attribute__ ((noinline)) __pgm_pow_scalar_dp_special_cases(__m128 res_exp, float const a_scalar, float const b_scalar)
{
   __m128 a = _mm_set1_ps(a_scalar);
   const __m128 b = _mm_set1_ps(b_scalar);
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


      return _mm_cvtss_f32(res_exp);
}

float __fss_pow_fma3(float const a_scalar, float const b_scalar)
{
//   fpminimax(log2(x),10,[|double...|],[0.5;0.9999999],relative);
   __m128d const LOG_C0_VEC   = _mm_set1_pd(LOG_C0);
   __m128d const LOG_C1_VEC   = _mm_set1_pd(LOG_C1);
   __m128d const LOG_C2_VEC   = _mm_set1_pd(LOG_C2);
   __m128d const LOG_C3_VEC   = _mm_set1_pd(LOG_C3);
   __m128d const LOG_C4_VEC   = _mm_set1_pd(LOG_C4);
   __m128d const LOG_C5_VEC   = _mm_set1_pd(LOG_C5);
   __m128d const LOG_C6_VEC   = _mm_set1_pd(LOG_C6);
   __m128d const LOG_C7_VEC   = _mm_set1_pd(LOG_C7);
   __m128d const LOG_C8_VEC   = _mm_set1_pd(LOG_C8);
   __m128d const LOG_C9_VEC   = _mm_set1_pd(LOG_C9);
   __m128d const LOG_C10_VEC  = _mm_set1_pd(LOG_C10);

//   fpminimax(exp(x*0.6931471805599453094172321214581765680755001343602552),6,[|double...|],[-0.5,0.5],relative);
   __m128d const EXP_C0_VEC = _mm_set1_pd(EXP_C0);
   __m128d const EXP_C1_VEC = _mm_set1_pd(EXP_C1);
   __m128d const EXP_C2_VEC = _mm_set1_pd(EXP_C2);
   __m128d const EXP_C3_VEC = _mm_set1_pd(EXP_C3);
   __m128d const EXP_C4_VEC = _mm_set1_pd(EXP_C4);
   __m128d const EXP_C5_VEC = _mm_set1_pd(EXP_C5);
   __m128d const EXP_C6_VEC = _mm_set1_pd(EXP_C6);

   __m128  const ONE_F_VEC = _mm_set1_ps(D_ONE_F);
   __m128  const ZERO_F_VEC = _mm_setzero_ps();
   __m128i const ALL_ONES_EXPONENT = _mm_set1_epi32(D_ALL_ONES_EXPONENT);

   __m128i const bit_mask2 = _mm_set1_epi32(D_BIT_MASK2);
   __m128i exp_offset = _mm_set1_epi32(D_EXP_OFFSET);
   __m128i const offset = _mm_set1_epi32(D_OFFSET);

   __m128d const EXP_HI_VEC = _mm_set1_pd(EXP_HI);
   __m128d const EXP_LO_VEC = _mm_set1_pd(EXP_LO);
   __m128d const DBL2INT_CVT_VEC= _mm_set1_pd(DBL2INT_CVT);

   __m128 const TWO_TO_M126_F_VEC = _mm_set1_ps(0x1p-126f);
   __m128i const U24_VEC = _mm_set1_epi32(D_U24);
   __m128 const TWO_TO_24_F_VEC = _mm_set1_ps(D_TWO_TO_24_F);
   __m128i sign_mask2 = _mm_set1_epi32(D_SIGN_MASK2);
   __m128 a = _mm_set1_ps(a_scalar);
   __m128 a_compute = _mm_and_ps(a, (__m128)sign_mask2);
   __m128 res;
   const __m128 b = _mm_set1_ps(b_scalar);

   __m128d b_d = _mm_cvtss_sd(b_d, b);

   __m128 mask = (__m128)_mm_cmplt_ps((__m128)a_compute, TWO_TO_M126_F_VEC);
#if defined(TARGET_LINUX_POWER)
   int moved_mask = _vec_any_nz((__m128i)mask);
#else
   int moved_mask = _mm_movemask_ps(mask);
#endif
   if (moved_mask) {
      a_compute= _mm_or_ps( _mm_and_ps(mask, _mm_mul_ps(a_compute, TWO_TO_24_F_VEC)), _mm_andnot_ps(mask,a_compute));
      exp_offset = _mm_add_epi32(exp_offset, _mm_and_si128((__m128i)mask, U24_VEC));
   }

   __m128i e_int = _mm_sub_epi32(_mm_srli_epi32( (__m128i)a_compute, 23), exp_offset);

   __m128d e = _mm_cvtepi32_pd(e_int);

   __m128 detect_inf_nan = _mm_add_ps(a_compute, b);
   __m128i overridemask = _mm_cmpeq_epi32( (__m128i)a_compute, (__m128i)ONE_F_VEC);
   overridemask = _mm_or_si128( overridemask, (__m128i)_mm_cmpeq_ps( b, ZERO_F_VEC));
   overridemask = _mm_or_si128( overridemask, _mm_cmpeq_epi32( _mm_and_si128((__m128i)detect_inf_nan, ALL_ONES_EXPONENT), (__m128i)ALL_ONES_EXPONENT));
   overridemask = _mm_or_si128( overridemask, (__m128i)_mm_cmple_ps(a, ZERO_F_VEC));
   int reducedMask = _mm_movemask_epi8(overridemask);

   __m128 m = (__m128)_mm_add_epi32(_mm_and_si128( (__m128i)a_compute, bit_mask2), offset);
   __m128d m_d = _mm_cvtss_sd(m_d, m);


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

   __m128d m2 = _mm_mul_sd(m_d, m_d);
   __m128d m4 = _mm_mul_sd(m2, m2);


   __m128d a1 = _mm_fmadd_sd(m_d, LOG_C9_VEC, LOG_C10_VEC);
   __m128d a2 = _mm_fmadd_sd(m_d, LOG_C7_VEC, LOG_C8_VEC);
   __m128d a3 = _mm_fmadd_sd(m_d, LOG_C5_VEC, LOG_C6_VEC);
   __m128d a4 = _mm_fmadd_sd(m_d, LOG_C3_VEC, LOG_C4_VEC);
   __m128d a5 = _mm_fmadd_sd(m_d, LOG_C1_VEC, LOG_C2_VEC);
   __m128d a6 = _mm_mul_sd(LOG_C0_VEC, m2);


   __m128d a7 = _mm_fmadd_sd(m2, a2, a1);
   __m128d a8 = _mm_fmadd_sd(m2, a4, a3);
   __m128d a9 = _mm_add_sd(a5, a6);

   __m128d a10 = _mm_fmadd_sd(m4, a9, a8);
   __m128d t = _mm_fmadd_sd(m4, a10, a7);

   t = _mm_add_sd(e, t);
   __m128d temp = _mm_mul_sd(b_d, t);

   //---------exponent starts here
//   __m256i exp_override = (__m256i)_mm256_cmp_pd( temp_hi, EXP_HI_VEC, _CMP_GT_OS);
//   exp_override = _mm256_or_si256(exp_override, (__m256i)_mm256_cmp_pd(temp_lo, EXP_HI_VEC, _CMP_GT_OS));
//   exp_override = _mm256_or_si256(exp_override, (__m256i)_mm256_cmp_pd(temp_hi, EXP_LO_VEC, _CMP_LT_OS));
//   exp_override = _mm256_or_si256(exp_override, (__m256i)_mm256_cmp_pd(temp_lo, EXP_LO_VEC, _CMP_LT_OS));
//   int exp_reduced_mask= _mm256_movemask_epi8(exp_override);
//   if (exp_reduced_mask) {
//      return pow_vec256_dp_slowpath(a, b);
//   }

   temp = _mm_min_sd(temp,EXP_HI_VEC );
   temp = _mm_max_sd(temp,EXP_LO_VEC );

   __m128d t_exp = _mm_add_sd(temp, DBL2INT_CVT_VEC);

   __m128d tt = _mm_sub_sd(t_exp, DBL2INT_CVT_VEC);
   __m128i integer = _mm_castpd_si128(t_exp);
   __m128d z_exp = _mm_sub_sd( temp, tt);

   __m128d poly_exp;
   poly_exp = EXP_C0_VEC;
   poly_exp = _mm_fmadd_sd(poly_exp, z_exp, EXP_C1_VEC);
   poly_exp = _mm_fmadd_sd(poly_exp, z_exp, EXP_C2_VEC);
   poly_exp = _mm_fmadd_sd(poly_exp, z_exp, EXP_C3_VEC);
   poly_exp = _mm_fmadd_sd(poly_exp, z_exp, EXP_C4_VEC);
   poly_exp = _mm_fmadd_sd(poly_exp, z_exp, EXP_C5_VEC);
   poly_exp = _mm_fmadd_sd(poly_exp, z_exp, EXP_C6_VEC);


//   __m128d b1 = _mm_fmadd_sd(z_exp, EXP_C5_VEC, EXP_C6_VEC);
//   __m128d b2 = _mm_fmadd_sd(z_exp, EXP_C3_VEC, EXP_C4_VEC);
//   __m128d z2_exp = _mm_mul_sd(z_exp, z_exp);
//   __m128d b3 = _mm_fmadd_sd(z_exp, EXP_C1_VEC, EXP_C2_VEC);

//   __m128d b5 = _mm_fmadd_sd(z2_exp, EXP_C0_VEC, b3);
//   __m128d b6 = _mm_fmadd_sd(z2_exp, b5, b2);
//   __m128d poly_exp = _mm_fmadd_sd(z2_exp, b6, b1);

   __m128i integer_poly_exp = _mm_castpd_si128(poly_exp);
   integer = _mm_slli_epi64(integer, 52);
   integer_poly_exp = _mm_add_epi32(integer, integer_poly_exp);
   __m128 res_exp = _mm_cvtpd_ps((__m128d)integer_poly_exp);

   if( __builtin_expect(reducedMask,0)) {
      return __pgm_pow_scalar_dp_special_cases(res_exp, a_scalar, b_scalar);
   }
   return _mm_cvtss_f32(res_exp);
}
