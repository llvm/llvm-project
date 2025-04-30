/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include <stdint.h>

#if defined(TARGET_LINUX_POWER)
#error "Source cannot be compiled for POWER architectures"
#include "xmm2altivec.h"
#else
#include <immintrin.h>
#include "mth_avx512helper.h"
#endif
#include "dpow_defs.h"

extern "C" __m512d FCN_AVX512(__fvd_pow_fma3)(__m512d, __m512d);


// struct representing a double-double number
// x, y represent the lo and hi parts respectively
struct __m512d_2 
{
    __m512d x, y;
};

// negates a number
inline
__m512d neg(__m512d a)
{
    __m512d const SGN_MASK = (__m512d)_mm512_set1_epi64(SGN_MASK_D);
    return _MM512_XOR_PD(a, SGN_MASK);
}

// double-double "fma"
inline
__m512d_2 __internal_ddfma (__m512d_2 x, __m512d_2 y, __m512d_2 z)
{
    __m512d e;
    __m512d_2 t, m;
    t.y = _mm512_mul_pd(x.y, y.y);
    t.x = _mm512_fmsub_pd(x.y, y.y, t.y);
    t.x = _mm512_fmadd_pd(x.y, y.x, t.x);
    t.x = _mm512_fmadd_pd(x.x, y.y, t.x);

    m.y = _mm512_add_pd (z.y, t.y);
    e = _mm512_sub_pd (z.y, m.y);
    m.x = _mm512_add_pd (_mm512_add_pd(_mm512_add_pd (e, t.y), t.x), z.x);

    return m;
}

// special case of double-double addition, where |x| > |y| and y is a double.
inline
__m512d_2 __internal_ddadd_yisdouble(__m512d_2 x, __m512d_2 y)
{
    __m512d_2 z;
    __m512d e;
    z.y = _mm512_add_pd (x.y, y.y);
    e = _mm512_sub_pd (x.y, z.y);
    z.x = _mm512_add_pd(_mm512_add_pd (e, y.y), x.x);

    return z;
}

// casts int to double
inline
__m512d __internal_fast_int2dbl(__m512i a)
{
    __m512i const INT2DBL_HI = _mm512_set1_epi64(INT2DBL_HI_D);
    __m512i const INT2DBL_LO = _mm512_set1_epi64(INT2DBL_LO_D);
    __m512d const INT2DBL    = (__m512d)_mm512_set1_epi64(INT2DBL_D);

    __m512i t = _mm512_xor_si512(INT2DBL_LO, a);
    t = _MM512_BLEND_EPI32(INT2DBL_HI, t, 0x5555);
    return _mm512_sub_pd((__m512d)t, INT2DBL);
}

// slowpath for exp. used to improve accuracy on larger inputs
static __m512d __attribute__ ((noinline)) __pgm_exp_d_vec512_slowpath(__m512i const i, __m512d const t, __m512d const bloga, __m512d z, __m512d prodx)
{
    __m512d const UPPERBOUND_1 = (__m512d)_mm512_set1_epi64(UPPERBOUND_1_D);
    __m512d const UPPERBOUND_2 = (__m512d)_mm512_set1_epi64(UPPERBOUND_2_D);
    __m512d const ZERO         = _mm512_set1_pd(ZERO_D);
    __m512d const INF          = (__m512d)_mm512_set1_epi64(INF_D);
    __m512i const HI_ABS_MASK  = _mm512_set1_epi64(HI_ABS_MASK_D);
    __m512i const ABS_MASK     = _mm512_set1_epi64(ABS_MASK_D);
    __m512i const MULT_CONST   = _mm512_set1_epi64(MULT_CONST_D);

    __m512i abs_bloga = _mm512_and_si512((__m512i)bloga, ABS_MASK);
    __m512d abs_lt = (__m512d)_mm512_and_si512((__m512i)abs_bloga, HI_ABS_MASK);
    __m512d lt_zero_mask = _MM512_CMP_PD(bloga, ZERO, _CMP_LT_OS); // compute bloga < 0.0  
    __m512d slowpath_mask = _MM512_CMP_PD(abs_lt, UPPERBOUND_1, _CMP_LT_OS); 

    __m512d a_plus_inf = _mm512_add_pd(bloga, INF); // check if bloga is too big      
    __m512d zero_inf_blend = _MM512_BLENDV_PD(a_plus_inf, ZERO, lt_zero_mask);     

    __m512d accurate_scale_mask = (__m512d)_MM512_CMP_PD(abs_lt, UPPERBOUND_2, 1); 

    // compute accurate scale
    __m512i k = _mm512_srli_epi64(i, 1); // k = i / 2                             
    __m512i i_scale_acc = _mm512_slli_epi64(k, D52_D);  // shift to HI and shift 20 
    k = _mm512_sub_epi32(i, k);          // k = i - k                            
    __m512i i_scale_acc_2 = _mm512_slli_epi64(k, D52_D);  // shift to HI and shift 20 
    __m512d multiplier = (__m512d)_mm512_add_epi64(i_scale_acc_2, MULT_CONST);
    multiplier = _MM512_BLENDV_PD(ZERO, multiplier, accurate_scale_mask); // quick fix for overflows in case they're being trapped

    __m512d res = (__m512d)_mm512_add_epi32(i_scale_acc, (__m512i)t);            
    res = _mm512_mul_pd(res, multiplier);                                       

    __m512d slowpath_blend = _MM512_BLENDV_PD(zero_inf_blend, res, accurate_scale_mask); 
    z = _MM512_BLENDV_PD(slowpath_blend, z, slowpath_mask);

    __m512i isinf_mask = _MM512_CMPEQ_EPI64((__m512i)INF, (__m512i)z); // special case for inf
    __m512d z_fixed = _mm512_fmadd_pd(z, prodx, z); // add only if not inf
    z = _MM512_BLENDV_PD(z_fixed, z, (__m512d)isinf_mask);

    return z;
}

// special cases for pow
// implementation derived from cudart/device_functions_impl.c
static __m512d __attribute__ ((noinline)) __pgm_pow_d_vec512_special_cases(__m512d const a, __m512d const b, __m512d t)
{
   __m512i const HI_MASK       = _mm512_set1_epi64(HI_MASK_D);
   __m512d const SGN_EXP_MASK  = (__m512d)_mm512_set1_epi64(SGN_EXP_MASK_D);
   __m512i const ABS_MASK      = _mm512_set1_epi64(ABS_MASK_D);
   __m512d const NEG_ONE       = _mm512_set1_pd(NEG_ONE_D);
   __m512d const ZERO          = _mm512_set1_pd(ZERO_D);
   __m512d const ONE           = _mm512_set1_pd(ONE_D);
   __m512d const HALF          = _mm512_set1_pd(HALF_D);
   __m512d const SGN_MASK      = (__m512d)_mm512_set1_epi64(SGN_MASK_D);
   __m512d const INF           = (__m512d)_mm512_set1_epi64(INF_D);
   __m512i const INF_FAKE      = _mm512_set1_epi64(INF_FAKE_D);
   __m512d const NAN_MASK      = (__m512d)_mm512_set1_epi64(NAN_MASK_D);
   __m512d const NEG_ONE_CONST = (__m512d)_mm512_set1_epi64(NEG_ONE_CONST_D);

   __m512i const TEN_23        = _mm512_set1_epi64(TEN_23_D);
   __m512i const ELEVEN        = _mm512_set1_epi64(ELEVEN_D);

    __m512i a_sign, b_sign, bIsOddInteger;
    __m512d abs_a, abs_b;
    __m512i shiftb;

    a_sign = (__m512i)_MM512_AND_PD(a, SGN_MASK);
    b_sign = (__m512i)_MM512_AND_PD(b, SGN_MASK);
    abs_a = _MM512_AND_PD(a, (__m512d)ABS_MASK); 
    abs_b = _MM512_AND_PD(b, (__m512d)ABS_MASK);

    // determining if b is an odd integer, since there are special cases for it
    shiftb = (__m512i)_MM512_AND_PD (b, SGN_EXP_MASK);
    shiftb = _mm512_srli_epi64(shiftb, D52_D);
    shiftb = _mm512_sub_epi64(shiftb, TEN_23);
    shiftb = _mm512_add_epi64(shiftb, ELEVEN);

    __m512i b_is_half = _MM512_CMPEQ_EPI64((__m512i)abs_b, (__m512i)HALF);
    bIsOddInteger = _mm512_sllv_epi64((__m512i)b, shiftb);
    bIsOddInteger = _MM512_CMPEQ_EPI64((__m512i)SGN_MASK, bIsOddInteger);
    // fix for b = +/-0.5 being incorrectly identified as an odd integer
    bIsOddInteger = _mm512_andnot_si512(b_is_half, bIsOddInteger);

    // corner cases where a <= 0
    // if ((ahi < 0) && bIsOddInteger)
    __m512d ahi_lt_0 = (__m512d)_MM512_CMPEQ_EPI64(a_sign, (__m512i)ZERO);
    __m512d ahilt0_and_boddint_mask = _MM512_ANDNOT_PD(ahi_lt_0, (__m512d)bIsOddInteger);

    t = _MM512_BLENDV_PD(t, neg(t), ahilt0_and_boddint_mask);

    // else if ((ahi < 0) && (b != trunc(b)))
    __m512d b_ne_trunc = _MM512_CMP_PD(b, _MM512_ROUND_PD(b, _MM_FROUND_TO_ZERO), _CMP_NEQ_UQ);
    __m512d nan_mask = _MM512_ANDNOT_PD(ahi_lt_0, b_ne_trunc);
   t = _MM512_BLENDV_PD(t, NAN_MASK, nan_mask);

    // if (a == 0.0)
    __m512d a_is_0_mask = (__m512d)_MM512_CMPEQ_EPI64((__m512i)abs_a, (__m512i)ZERO);
    __m512d thi_when_ais0 = ZERO;

    // if (bIsOddInteger && a == 0.0)
    thi_when_ais0 = _MM512_BLENDV_PD(thi_when_ais0, (__m512d)a, (__m512d)bIsOddInteger);

    // if (bhi < 0 && a == 0.0)
    __m512d bhi_lt_0 = (__m512d)_MM512_CMPEQ_EPI64(b_sign, (__m512i)ZERO); //this mask is inverted
    __m512d thi_or_INF = _MM512_OR_PD(thi_when_ais0, INF);
    thi_when_ais0 = _MM512_BLENDV_PD(thi_or_INF, thi_when_ais0, bhi_lt_0);
    __m512d t_when_ais0 = _MM512_AND_PD(thi_when_ais0, (__m512d)HI_MASK);

    t = _MM512_BLENDV_PD(t, t_when_ais0, a_is_0_mask);

    // else if (a is INF)
    __m512d a_inf_mask = (__m512d)_MM512_CMPEQ_EPI64((__m512i)abs_a, (__m512i)INF);
    // use bhi_lt_0 backwards to evaluate bhi >= 0
    __m512d thi_when_aisinf = _MM512_BLENDV_PD(thi_when_aisinf, INF, bhi_lt_0);

    // now evaluate ((ahi < 0) && bIsOddInteger)
    __m512d thi_xor_sgn = _MM512_XOR_PD(thi_when_aisinf, SGN_MASK);
    thi_when_aisinf = _MM512_BLENDV_PD(thi_when_aisinf, thi_xor_sgn, ahilt0_and_boddint_mask);

    t = _MM512_BLENDV_PD(t, thi_when_aisinf, a_inf_mask);

    // else if (abs(b) is INF)
    __m512d b_inf_mask = (__m512d)_MM512_CMPEQ_EPI64((__m512i)abs_b, (__m512i)INF);
    __m512d thi_when_bisinf = ZERO;
    __m512d absa_gt_one = _MM512_CMP_PD(abs_a, ONE, _CMP_GT_OS); // evaluating (abs(a) > 1)
    thi_when_bisinf = _MM512_BLENDV_PD(thi_when_bisinf, INF, absa_gt_one);

    __m512d thi_xor_inf = _MM512_XOR_PD(thi_when_bisinf, INF);
    thi_when_bisinf = _MM512_BLENDV_PD(thi_xor_inf, thi_when_bisinf, bhi_lt_0); // bhi < 0

    __m512d a_is_negone = (__m512d)_MM512_CMPEQ_EPI64((__m512i)a, (__m512i)NEG_ONE);
    thi_when_bisinf = _MM512_BLENDV_PD(thi_when_bisinf, NEG_ONE_CONST, a_is_negone); //a == -1

    t = _MM512_BLENDV_PD(t, thi_when_bisinf, b_inf_mask);

    // if(a is NAN || B is NAN) <=> !(a is a number && b is a number)
    __m512i a_nan_mask = _MM512_CMPEQ_EPI64(_mm512_max_epu32((__m512i)abs_a, INF_FAKE), INF_FAKE);
    __m512i b_nan_mask = _MM512_CMPEQ_EPI64(_mm512_max_epu32((__m512i)abs_b, INF_FAKE), INF_FAKE);
    __m512i aorb_nan_mask = _mm512_and_si512(a_nan_mask, b_nan_mask); //this mask is inverted
    t = _MM512_BLENDV_PD(_mm512_add_pd(a,b), t, (__m512d)aorb_nan_mask);

    // if a == 1 or b == 0, answer = 1 
    __m512d a_is_one_mask = (__m512d)_MM512_CMPEQ_EPI64((__m512i)a, (__m512i)ONE);
    __m512d b_is_zero_mask = (__m512d)_MM512_CMPEQ_EPI64((__m512i)abs_b, (__m512i)ZERO);
    __m512d ans_equals_one_mask = _MM512_OR_PD(a_is_one_mask, b_is_zero_mask);
    t = _MM512_BLENDV_PD(t, ONE, ans_equals_one_mask);
    // ****************************************************************************************** 
 

   __m512i a_is_neg = (__m512i)_MM512_CMP_PD(a, ZERO, _CMP_LT_OS);
   int a_is_neg_flag = _MM512_MOVEMASK_PD(_mm512_castsi512_pd(a_is_neg));

   __m512i b_is_integer = (__m512i)_MM512_CMP_PD(b, _mm512_floor_pd(b), _CMP_EQ_OQ);
   int b_is_integer_flag = _MM512_MOVEMASK_PD(_mm512_castsi512_pd(b_is_integer));

   __m512i b_is_lt_zero = (__m512i)_MM512_CMP_PD(b, ZERO, _CMP_LT_OS);
   int b_is_lt_zero_flag = _MM512_MOVEMASK_PD(_mm512_castsi512_pd(b_is_lt_zero));

   __m512d const MINUS_ZERO = _mm512_set1_pd((double)-0.0);
   __m512i a_is_pos_zero = (__m512i) _MM512_CMPEQ_PD( a, ZERO);
   __m512i a_is_neg_zero = (__m512i) _MM512_CMPEQ_PD( a, MINUS_ZERO);
   __m512i a_is_any_zero = _mm512_or_si512(a_is_pos_zero, a_is_neg_zero);
   int a_is_any_zero_flag = _MM512_MOVEMASK_PD(_mm512_castsi512_pd(a_is_any_zero));

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
         __m512d volatile invop = _mm512_sqrt_pd(a);
      }

      if (a_is_any_zero_flag && b_is_lt_zero_flag) {
         __m512d volatile divXzero = _mm512_div_pd(ONE,ZERO);
      }











 
    return t;
}

__m512d FCN_AVX512(__fvd_pow_fma3)(__m512d const a, __m512d const b)
{
    //////////////////////////////////////////////////// log constants
    __m512d const LOG_POLY_6 = _mm512_set1_pd(LOG_POLY_6_D);
    __m512d const LOG_POLY_5 = _mm512_set1_pd(LOG_POLY_5_D);
    __m512d const LOG_POLY_4 = _mm512_set1_pd(LOG_POLY_4_D);
    __m512d const LOG_POLY_3 = _mm512_set1_pd(LOG_POLY_3_D);
    __m512d const LOG_POLY_2 = _mm512_set1_pd(LOG_POLY_2_D);
    __m512d const LOG_POLY_1 = _mm512_set1_pd(LOG_POLY_1_D);
    __m512d const LOG_POLY_0 = _mm512_set1_pd(LOG_POLY_0_D);

    __m512d const CC_CONST_Y   = _mm512_set1_pd(CC_CONST_Y_D);
    __m512d const CC_CONST_X   = _mm512_set1_pd(CC_CONST_X_D);

    __m512i const EXPO_MASK    = _mm512_set1_epi64(EXPO_MASK_D);
    __m512d const HI_CONST_1   = (__m512d)_mm512_set1_epi64(HI_CONST_1_D);
    __m512d const HI_CONST_2   = (__m512d)_mm512_set1_epi64(HI_CONST_2_D);
    __m512i const HALFIFIER    = _mm512_set1_epi64(HALFIFIER_D); 
    __m512i const HI_THRESH    = _mm512_set1_epi64(HI_THRESH_D); //~sqrt(2)
    __m512d const ONE_F        = _mm512_set1_pd(ONE_F_D);
    __m512d const TWO          = _mm512_set1_pd(TWO_D);
    __m512d const ZERO         = _mm512_set1_pd(ZERO_D);

    __m512d const LN2_HI       = _mm512_set1_pd(LN2_HI_D);
    __m512d const LN2_LO       = _mm512_set1_pd(LN2_LO_D);
    //////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////// exp constants
    __m512d const L2E        = _mm512_set1_pd(L2E_D);
    __m512d const NEG_LN2_HI = _mm512_set1_pd(NEG_LN2_HI_D);
    __m512d const NEG_LN2_LO = _mm512_set1_pd(NEG_LN2_LO_D);

    __m512d const EXP_POLY_B = _mm512_set1_pd(EXP_POLY_B_D);
    __m512d const EXP_POLY_A = _mm512_set1_pd(EXP_POLY_A_D);
    __m512d const EXP_POLY_9 = _mm512_set1_pd(EXP_POLY_9_D);
    __m512d const EXP_POLY_8 = _mm512_set1_pd(EXP_POLY_8_D);
    __m512d const EXP_POLY_7 = _mm512_set1_pd(EXP_POLY_7_D);
    __m512d const EXP_POLY_6 = _mm512_set1_pd(EXP_POLY_6_D);
    __m512d const EXP_POLY_5 = _mm512_set1_pd(EXP_POLY_5_D);
    __m512d const EXP_POLY_4 = _mm512_set1_pd(EXP_POLY_4_D);
    __m512d const EXP_POLY_3 = _mm512_set1_pd(EXP_POLY_3_D);
    __m512d const EXP_POLY_2 = _mm512_set1_pd(EXP_POLY_2_D);
    __m512d const EXP_POLY_1 = _mm512_set1_pd(EXP_POLY_1_D);
    __m512d const EXP_POLY_0 = _mm512_set1_pd(EXP_POLY_0_D);

    __m512d const DBL2INT_CVT = _mm512_set1_pd(DBL2INT_CVT_D);

    __m512d const UPPERBOUND_1 = (__m512d)_mm512_set1_epi64(UPPERBOUND_1_D);
    __m512i const HI_ABS_MASK = _mm512_set1_epi64(HI_ABS_MASK_D);

    //////////////////////////////////////////////////// pow constants
    __m512i const HI_MASK      = _mm512_set1_epi64(HI_MASK_D);
    __m512d const SGN_EXP_MASK = (__m512d)_mm512_set1_epi64(SGN_EXP_MASK_D);
    __m512i const ABS_MASK     = _mm512_set1_epi64(ABS_MASK_D);
    __m512i const ONE         = _mm512_set1_epi64(ONE_D);

    __m512i const TEN_23      = _mm512_set1_epi64(TEN_23_D);
    __m512i const ALL_ONES_EXPONENT = _mm512_set1_epi64(ALL_ONES_EXPONENT_D);


    __m512d abs_a, a_mut;
    __m512d_2 loga;
    __m512d t_hi, t_lo, tmp, e, bloga, prodx;

    __m512d_2 qq, cc, uu, tt;
    __m512d f, g, u, v, q, ulo, m;
    __m512i ihi, expo, expo_plus1;
    __m512d thresh_mask;
    __m512d b_is_one_mask;

    /*
     * Check whether all exponents in vector b are 1.0, and if so take
     * a quick exit.
     * Note: b_is_one_mask is needed later in case some elements in b are 1.0.
     */

    b_is_one_mask = _MM512_CMP_PD(b, ONE_F, _CMP_EQ_OQ);
//Correct - but slower
//    if (_MM512_MOVEMASK_PD(b_is_one_mask) == 0xff) {
    if (_mm512_cmp_pd_mask(b, ONE_F, _CMP_EQ_OQ) == 0xff) {
        return a;
    }

    // *****************************************************************************************
    // computing log(abs(a))
    abs_a = _MM512_AND_PD(a, (__m512d)ABS_MASK);
    a_mut = _MM512_AND_PD(abs_a, HI_CONST_1);
    a_mut = _MM512_OR_PD(a_mut, HI_CONST_2);
    m = (__m512d)_mm512_sub_epi32((__m512i)a_mut, HALFIFIER); // divide by 2 

    ihi = _mm512_and_si512((__m512i)a_mut, HI_MASK);
    thresh_mask = _MM512_CMP_PD((__m512d)ihi, (__m512d)HI_THRESH,_CMP_GT_OS); 
    m = _MM512_BLENDV_PD(a_mut, m, thresh_mask); 

    expo = _mm512_srli_epi64((__m512i)abs_a, D52_D);
    expo = _mm512_sub_epi64(expo, TEN_23);
    expo_plus1 = _mm512_add_epi64(expo, ONE); // add one to exponent instead
    expo = (__m512i)_MM512_BLENDV_PD((__m512d)expo, (__m512d)expo_plus1, thresh_mask);

    // begin computing log(m)
    f = _mm512_sub_pd(m, ONE_F);
    g = _mm512_add_pd(m, ONE_F);
    g = _mm512_div_pd(ONE_F, g); 

    u = _mm512_mul_pd(_mm512_mul_pd(TWO, f), g);

    // u = 2.0 * (m - 1.0) / (m + 1.0) 
    v = _mm512_mul_pd(u, u);

    // polynomial is used to approximate atanh(v)
    // an estrin evaluation scheme is used.
    __m512d c0 = _mm512_fmadd_pd(LOG_POLY_1, v, LOG_POLY_0);
    __m512d c2 = _mm512_fmadd_pd(LOG_POLY_3, v, LOG_POLY_2);
    __m512d c4 = _mm512_fmadd_pd(LOG_POLY_5, v, LOG_POLY_4);
    __m512d v2 = _mm512_mul_pd(v, v);
    __m512d v4 = _mm512_mul_pd(v2, v2);

    c0 = _mm512_fmadd_pd(c2, v2, c0);
    c4 = _mm512_fmadd_pd(LOG_POLY_6, v2, c4);
    q = _mm512_fmadd_pd(c4, v4, c0);
    q = _mm512_mul_pd(q, v);

    tmp = _mm512_mul_pd(TWO, _mm512_sub_pd(f, u));
    tmp = _mm512_fmadd_pd(neg(u), f, tmp);
    ulo = _mm512_mul_pd(g, tmp);

    // double-double computation begins
    qq.y = q;
    qq.x = ZERO;
    uu.y = u;
    uu.x = ulo;
    cc.y = CC_CONST_Y;
    cc.x = CC_CONST_X;

    qq = __internal_ddadd_yisdouble(cc, qq);

    // computing log(m) in double-double format
    cc.y = _mm512_mul_pd(uu.y, uu.y);
    cc.x = _mm512_fmsub_pd(uu.y, uu.y, cc.y);
    cc.x = _mm512_fmadd_pd(uu.y,
                           (__m512d)_mm512_add_epi32((__m512i)uu.x, HALFIFIER), 
                           cc.x); // u ** 2

    tt.y = _mm512_mul_pd(cc.y, uu.y);
    tt.x = _mm512_fmsub_pd(cc.y, uu.y, tt.y);
    tt.x = _mm512_fmadd_pd(cc.y, uu.x, tt.x);
    tt.x = _mm512_fmadd_pd(cc.x, uu.y, tt.x); // u ** 3 un-normalized

    uu = __internal_ddfma(qq, tt, uu);

    // computing log a = log(m) + log(2)*expo
    f = __internal_fast_int2dbl (expo);
    q = _mm512_fmadd_pd(f, LN2_HI, uu.y);
    tmp = _mm512_fmadd_pd(neg(f), LN2_HI, q);
    tmp = _mm512_sub_pd(tmp, uu.y);
    loga.y = q;
    loga.x = _mm512_sub_pd(uu.x, tmp);
    loga.x = _mm512_fmadd_pd(f, LN2_LO, loga.x);

    // finish log(a)
    // ******************************************************************************************
    
    // compute b * log(a)
    t_hi = _mm512_mul_pd(loga.y, b);
    t_lo = _mm512_fmsub_pd(loga.y, b, t_hi);
    t_lo = _mm512_fmadd_pd(loga.x, b, t_lo);
    bloga = e = _mm512_add_pd(t_hi, t_lo);
    prodx = _mm512_add_pd(_mm512_sub_pd(t_hi, e), t_lo);

    // ***************************************************************************************
    // computing exp(b * log(a))
    // calculating exponent; stored in the LO of each 64-bit block
    __m512i i = (__m512i) _mm512_fmadd_pd(bloga, L2E, DBL2INT_CVT);

    // calculate mantissa
    __m512d t = _mm512_sub_pd ((__m512d)i, DBL2INT_CVT);
    __m512d z = _mm512_fmadd_pd (t, NEG_LN2_HI, bloga);
    z = _mm512_fmadd_pd (t, NEG_LN2_LO, z);

    // use polynomial to calculate exp
    // mixed estrin/horner scheme: estrin on the higher 8 coefficients, horner on the lowest 4.
    // provided speedup without loss of precision compared to full horner
    __m512d t4 = _mm512_fmadd_pd(EXP_POLY_5, z, EXP_POLY_4);
    __m512d t6 = _mm512_fmadd_pd(EXP_POLY_7, z, EXP_POLY_6);
    __m512d t8 = _mm512_fmadd_pd(EXP_POLY_9, z, EXP_POLY_8);
    __m512d t10 = _mm512_fmadd_pd(EXP_POLY_B, z, EXP_POLY_A);

    __m512d z2 = _mm512_mul_pd(z, z);
    __m512d z4 = _mm512_mul_pd(z2, z2);

    t4 = _mm512_fmadd_pd(t6, z2, t4);
    t8 = _mm512_fmadd_pd(t10, z2, t8);
    t4 = _mm512_fmadd_pd(t8, z4, t4);
    
    t = _mm512_fmadd_pd(t4, z, EXP_POLY_3);
    t = _mm512_fmadd_pd(t, z, EXP_POLY_2);
    t = _mm512_fmadd_pd(t, z, EXP_POLY_1);
    t = _mm512_fmadd_pd(t, z, EXP_POLY_0); 

    // fast scale
    __m512i i_scale = _mm512_slli_epi64(i,D52_D); 
    __m512d ztemp = z = (__m512d)_mm512_add_epi32(i_scale, (__m512i)t);  

    // slowpath detection for exp
    __m512d abs_bloga = (__m512d)_mm512_and_si512((__m512i)bloga, HI_ABS_MASK); 
    int exp_slowmask = _MM512_MOVEMASK_PD(_MM512_CMP_PD(abs_bloga, UPPERBOUND_1, _CMP_GE_OS));

    z = _mm512_fmadd_pd(z, prodx, z);     

    if (__builtin_expect(exp_slowmask, 0)) {
        z = __pgm_exp_d_vec512_slowpath(i, t, bloga, ztemp, prodx); 
    }
    // finished exp(b * log (a))
    // ************************************************************************************************

    // Now special case if some elements of exponent (b) are 1.0.
    z = _MM512_BLENDV_PD(z, a, b_is_one_mask);

    // compute if we have special cases (inf, nan, etc). see man pow for full list of special cases
    __m512i detect_inf_nan = (__m512i)_mm512_add_pd(a, b);  // check for inf/nan
    __m512i overridemask = _MM512_CMPEQ_EPI64( (__m512i)a, (__m512i)ONE_F); //  if a == 1
    __m512i overridemask2 = _MM512_CMPEQ_EPI64(_mm512_and_si512(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);
    overridemask = _mm512_or_si512(overridemask, (__m512i)_MM512_CMP_PD(a, ZERO, _CMP_LE_OQ)); // if a < 0
         
    int specMask = _MM512_MOVEMASK_PD((__m512d)_mm512_or_si512(overridemask, overridemask2));
    if(__builtin_expect(specMask, 0)) {
        return __pgm_pow_d_vec512_special_cases(a, b, z);
    }
    return z;
}
     
