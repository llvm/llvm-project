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
#endif
#include "dpow_defs.h"

extern "C" __m256d __fvd_pow_fma3_256(__m256d, __m256d);


// struct representing a double-double number
// x, y represent the lo and hi parts respectively
struct __m256d_2 
{
    __m256d x, y;
};

// negates a number
inline
__m256d neg(__m256d a)
{
    __m256d const SGN_MASK = (__m256d)_mm256_set1_epi64x(SGN_MASK_D);
    return _mm256_xor_pd(a, SGN_MASK);
}

// double-double "fma"
inline
__m256d_2 __internal_ddfma (__m256d_2 x, __m256d_2 y, __m256d_2 z)
{
    __m256d e;
    __m256d_2 t, m;
    t.y = _mm256_mul_pd(x.y, y.y);
    t.x = _mm256_fmsub_pd(x.y, y.y, t.y);
    t.x = _mm256_fmadd_pd(x.y, y.x, t.x);
    t.x = _mm256_fmadd_pd(x.x, y.y, t.x);

    m.y = _mm256_add_pd (z.y, t.y);
    e = _mm256_sub_pd (z.y, m.y);
    m.x = _mm256_add_pd (_mm256_add_pd(_mm256_add_pd (e, t.y), t.x), z.x);

    return m;
}

// special case of double-double addition, where |x| > |y| and y is a double.
inline
__m256d_2 __internal_ddadd_yisdouble(__m256d_2 x, __m256d_2 y)
{
    __m256d_2 z;
    __m256d e;
    z.y = _mm256_add_pd (x.y, y.y);
    e = _mm256_sub_pd (x.y, z.y);
    z.x = _mm256_add_pd(_mm256_add_pd (e, y.y), x.x);

    return z;
}

// casts int to double
inline
__m256d __internal_fast_int2dbl(__m256i a)
{
    __m256i const INT2DBL_HI = _mm256_set1_epi64x(INT2DBL_HI_D);
    __m256i const INT2DBL_LO = _mm256_set1_epi64x(INT2DBL_LO_D);
    __m256d const INT2DBL    = (__m256d)_mm256_set1_epi64x(INT2DBL_D);

    __m256i t = _mm256_xor_si256(INT2DBL_LO, a);
    t = _mm256_blend_epi32(INT2DBL_HI, t, 0x55); 
    return _mm256_sub_pd((__m256d)t, INT2DBL);
}

// slowpath for exp. used to improve accuracy on larger inputs
__m256d __attribute__ ((noinline)) __pgm_exp_d_vec256_slowpath(__m256i const i, __m256d const t, __m256d const bloga, __m256d z, __m256d prodx)
{
    __m256d const UPPERBOUND_1 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_1_D);
    __m256d const UPPERBOUND_2 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_2_D);
    __m256d const ZERO         = _mm256_set1_pd(ZERO_D);
    __m256d const INF          = (__m256d)_mm256_set1_epi64x(INF_D);
    __m256i const HI_ABS_MASK  = _mm256_set1_epi64x(HI_ABS_MASK_D);
    __m256i const ABS_MASK     = _mm256_set1_epi64x(ABS_MASK_D);
    __m256i const MULT_CONST   = _mm256_set1_epi64x(MULT_CONST_D);

    __m256i abs_bloga = _mm256_and_si256((__m256i)bloga, ABS_MASK);
    __m256d abs_lt = (__m256d)_mm256_and_si256((__m256i)abs_bloga, HI_ABS_MASK);
    __m256d lt_zero_mask = _mm256_cmp_pd(bloga, ZERO, _CMP_LT_OS); // compute bloga < 0.0  
    __m256d slowpath_mask = _mm256_cmp_pd(abs_lt, UPPERBOUND_1, _CMP_LT_OS); 

    __m256d a_plus_inf = _mm256_add_pd(bloga, INF); // check if bloga is too big      
    __m256d zero_inf_blend = _mm256_blendv_pd(a_plus_inf, ZERO, lt_zero_mask);     

    __m256d accurate_scale_mask = (__m256d)_mm256_cmp_pd(abs_lt, UPPERBOUND_2, 1); 

    // compute accurate scale
    __m256i k = _mm256_srli_epi64(i, 1); // k = i / 2                             
    __m256i i_scale_acc = _mm256_slli_epi64(k, D52_D);  // shift to HI and shift 20 
    k = _mm256_sub_epi32(i, k);          // k = i - k                            
    __m256i i_scale_acc_2 = _mm256_slli_epi64(k, D52_D);  // shift to HI and shift 20 
    __m256d multiplier = (__m256d)_mm256_add_epi64(i_scale_acc_2, MULT_CONST);
    multiplier = _mm256_blendv_pd(ZERO, multiplier, accurate_scale_mask); // quick fix for overflows in case they're being trapped

    __m256d res = (__m256d)_mm256_add_epi32(i_scale_acc, (__m256i)t);            
    res = _mm256_mul_pd(res, multiplier);                                       

    __m256d slowpath_blend = _mm256_blendv_pd(zero_inf_blend, res, accurate_scale_mask); 
    z = _mm256_blendv_pd(slowpath_blend, z, slowpath_mask);

    __m256i isinf_mask = _mm256_cmpeq_epi64((__m256i)INF, (__m256i)z); // special case for inf
    __m256d z_fixed = _mm256_fmadd_pd(z, prodx, z); // add only if not inf
    z = _mm256_blendv_pd(z_fixed, z, (__m256d)isinf_mask);

    return z;
}

// special cases for pow
// implementation derived from cudart/device_functions_impl.c
__m256d __attribute__ ((noinline)) __pgm_pow_d_vec256_special_cases(__m256d const a, __m256d const b, __m256d t)
{
   __m256i const HI_MASK       = _mm256_set1_epi64x(HI_MASK_D);
   __m256d const SGN_EXP_MASK  = (__m256d)_mm256_set1_epi64x(SGN_EXP_MASK_D);
   __m256i const ABS_MASK      = _mm256_set1_epi64x(ABS_MASK_D);
   __m256d const NEG_ONE       = _mm256_set1_pd(NEG_ONE_D);
   __m256d const ZERO          = _mm256_set1_pd(ZERO_D);
   __m256d const ONE           = _mm256_set1_pd(ONE_D);
   __m256d const HALF          = _mm256_set1_pd(HALF_D);
   __m256d const SGN_MASK      = (__m256d)_mm256_set1_epi64x(SGN_MASK_D);
   __m256d const INF           = (__m256d)_mm256_set1_epi64x(INF_D);
   __m256i const INF_FAKE      = _mm256_set1_epi64x(INF_FAKE_D);
   __m256d const NAN_MASK      = (__m256d)_mm256_set1_epi64x(NAN_MASK_D);
   __m256d const NEG_ONE_CONST = (__m256d)_mm256_set1_epi64x(NEG_ONE_CONST_D);

   __m256i const TEN_23        = _mm256_set1_epi64x(TEN_23_D);
   __m256i const ELEVEN        = _mm256_set1_epi64x(ELEVEN_D);

    __m256i a_sign, b_sign, bIsOddInteger;
    __m256d abs_a, abs_b;
    __m256i shiftb;

    a_sign = (__m256i)_mm256_and_pd(a, SGN_MASK);
    b_sign = (__m256i)_mm256_and_pd(b, SGN_MASK);
    abs_a = _mm256_and_pd(a, (__m256d)ABS_MASK); 
    abs_b = _mm256_and_pd(b, (__m256d)ABS_MASK);

    // determining if b is an odd integer, since there are special cases for it
    shiftb = (__m256i)_mm256_and_pd (b, SGN_EXP_MASK);
    shiftb = _mm256_srli_epi64(shiftb, D52_D);
    shiftb = _mm256_sub_epi64(shiftb, TEN_23);
    shiftb = _mm256_add_epi64(shiftb, ELEVEN);

    __m256i b_is_half = _mm256_cmpeq_epi64((__m256i)abs_b, (__m256i)HALF);
    bIsOddInteger = _mm256_sllv_epi64((__m256i)b, shiftb);
    bIsOddInteger = _mm256_cmpeq_epi64((__m256i)SGN_MASK, bIsOddInteger);
    // fix for b = +/-0.5 being incorrectly identified as an odd integer
    bIsOddInteger = _mm256_andnot_si256(b_is_half, bIsOddInteger);

    // corner cases where a <= 0
    // if ((ahi < 0) && bIsOddInteger)
    __m256d ahi_lt_0 = (__m256d)_mm256_cmpeq_epi64(a_sign, (__m256i)ZERO);
    __m256d ahilt0_and_boddint_mask = _mm256_andnot_pd(ahi_lt_0, (__m256d)bIsOddInteger);

    t = _mm256_blendv_pd(t, neg(t), ahilt0_and_boddint_mask);

    // else if ((ahi < 0) && (b != trunc(b)))
    __m256d b_ne_trunc = _mm256_cmp_pd(b, _mm256_round_pd(b, _MM_FROUND_TO_ZERO), _CMP_NEQ_UQ);
    __m256d nan_mask = _mm256_andnot_pd(ahi_lt_0, b_ne_trunc);
    t = _mm256_blendv_pd(t, NAN_MASK, nan_mask);

    // if (a == 0.0)
    __m256d a_is_0_mask = (__m256d)_mm256_cmpeq_epi64((__m256i)abs_a, (__m256i)ZERO);
    __m256d thi_when_ais0 = ZERO;

    // if (bIsOddInteger && a == 0.0)
    thi_when_ais0 = _mm256_blendv_pd(thi_when_ais0, (__m256d)a, (__m256d)bIsOddInteger);

    // if (bhi < 0 && a == 0.0)
    __m256d bhi_lt_0 = (__m256d)_mm256_cmpeq_epi64(b_sign, (__m256i)ZERO); //this mask is inverted
    __m256d thi_or_INF = _mm256_or_pd(thi_when_ais0, INF);
    thi_when_ais0 = _mm256_blendv_pd(thi_or_INF, thi_when_ais0, bhi_lt_0);
    __m256d t_when_ais0 = _mm256_and_pd(thi_when_ais0, (__m256d)HI_MASK);

    t = _mm256_blendv_pd(t, t_when_ais0, a_is_0_mask);

    // else if (a is INF)
    __m256d a_inf_mask = (__m256d)_mm256_cmpeq_epi64((__m256i)abs_a, (__m256i)INF);
    // use bhi_lt_0 backwards to evaluate bhi >= 0
    __m256d thi_when_aisinf = _mm256_blendv_pd(thi_when_aisinf, INF, bhi_lt_0);

    // now evaluate ((ahi < 0) && bIsOddInteger)
    __m256d thi_xor_sgn = _mm256_xor_pd(thi_when_aisinf, SGN_MASK);
    thi_when_aisinf = _mm256_blendv_pd(thi_when_aisinf, thi_xor_sgn, ahilt0_and_boddint_mask);

    t = _mm256_blendv_pd(t, thi_when_aisinf, a_inf_mask);

    // else if (abs(b) is INF)
    __m256d b_inf_mask = (__m256d)_mm256_cmpeq_epi64((__m256i)abs_b, (__m256i)INF);
    __m256d thi_when_bisinf = ZERO;
    __m256d absa_gt_one = _mm256_cmp_pd(abs_a, ONE, _CMP_GT_OS); // evaluating (abs(a) > 1)
    thi_when_bisinf = _mm256_blendv_pd(thi_when_bisinf, INF, absa_gt_one);

    __m256d thi_xor_inf = _mm256_xor_pd(thi_when_bisinf, INF);
    thi_when_bisinf = _mm256_blendv_pd(thi_xor_inf, thi_when_bisinf, bhi_lt_0); // bhi < 0

    __m256d a_is_negone = (__m256d)_mm256_cmpeq_epi64((__m256i)a, (__m256i)NEG_ONE);
    thi_when_bisinf = _mm256_blendv_pd(thi_when_bisinf, NEG_ONE_CONST, a_is_negone); //a == -1

    t = _mm256_blendv_pd(t, thi_when_bisinf, b_inf_mask);

    // if(a is NAN || B is NAN) <=> !(a is a number && b is a number)
    __m256i a_nan_mask = _mm256_cmpeq_epi64(_mm256_max_epu32((__m256i)abs_a, INF_FAKE), INF_FAKE);
    __m256i b_nan_mask = _mm256_cmpeq_epi64(_mm256_max_epu32((__m256i)abs_b, INF_FAKE), INF_FAKE);
    __m256i aorb_nan_mask = _mm256_and_si256(a_nan_mask, b_nan_mask); //this mask is inverted
    t = _mm256_blendv_pd(_mm256_add_pd(a,b), t, (__m256d)aorb_nan_mask);

    // if a == 1 or b == 0, answer = 1 
    __m256d a_is_one_mask = (__m256d)_mm256_cmpeq_epi64((__m256i)a, (__m256i)ONE);
    __m256d b_is_zero_mask = (__m256d)_mm256_cmpeq_epi64((__m256i)abs_b, (__m256i)ZERO);
    __m256d ans_equals_one_mask = _mm256_or_pd(a_is_one_mask, b_is_zero_mask);
    t = _mm256_blendv_pd(t, ONE, ans_equals_one_mask);
    // ****************************************************************************************** 
 

   __m256i a_is_neg = (__m256i)_mm256_cmp_pd(a, ZERO, _CMP_LT_OS);
   int a_is_neg_flag = _mm256_movemask_epi8((__m256i)a_is_neg);

   __m256i b_is_integer = (__m256i)_mm256_cmp_pd(b, _mm256_floor_pd(b), _CMP_EQ_OQ);
   int b_is_integer_flag = _mm256_movemask_epi8((__m256i)b_is_integer);

   __m256i b_is_lt_zero = (__m256i)_mm256_cmp_pd(b, ZERO, _CMP_LT_OS);
   int b_is_lt_zero_flag = _mm256_movemask_epi8((__m256i)b_is_lt_zero);

   __m256d const MINUS_ZERO = _mm256_set1_pd((double)-0.0);
   __m256i a_is_pos_zero = _mm256_cmpeq_epi64( (__m256i)a, (__m256i)ZERO);
   __m256i a_is_neg_zero = _mm256_cmpeq_epi64( (__m256i)a, (__m256i)MINUS_ZERO);
   __m256i a_is_any_zero = _mm256_or_si256(a_is_pos_zero, a_is_neg_zero);
   int a_is_any_zero_flag = _mm256_movemask_epi8((__m256i)a_is_any_zero);

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
         __m256d volatile invop = _mm256_sqrt_pd(a);
      }

      if (a_is_any_zero_flag && b_is_lt_zero_flag) {
         __m256d volatile divXzero = _mm256_div_pd(ONE,ZERO);
      }











 
    return t;
}

__m256d __fvd_pow_fma3_256(__m256d const a, __m256d const b)
{
    //////////////////////////////////////////////////// log constants
    __m256d const LOG_POLY_6 = _mm256_set1_pd(LOG_POLY_6_D);
    __m256d const LOG_POLY_5 = _mm256_set1_pd(LOG_POLY_5_D);
    __m256d const LOG_POLY_4 = _mm256_set1_pd(LOG_POLY_4_D);
    __m256d const LOG_POLY_3 = _mm256_set1_pd(LOG_POLY_3_D);
    __m256d const LOG_POLY_2 = _mm256_set1_pd(LOG_POLY_2_D);
    __m256d const LOG_POLY_1 = _mm256_set1_pd(LOG_POLY_1_D);
    __m256d const LOG_POLY_0 = _mm256_set1_pd(LOG_POLY_0_D);

    __m256d const CC_CONST_Y   = _mm256_set1_pd(CC_CONST_Y_D);
    __m256d const CC_CONST_X   = _mm256_set1_pd(CC_CONST_X_D);

    __m256i const EXPO_MASK    = _mm256_set1_epi64x(EXPO_MASK_D);
    __m256d const HI_CONST_1   = (__m256d)_mm256_set1_epi64x(HI_CONST_1_D);
    __m256d const HI_CONST_2   = (__m256d)_mm256_set1_epi64x(HI_CONST_2_D);
    __m256i const HALFIFIER    = _mm256_set1_epi64x(HALFIFIER_D); 
    __m256i const HI_THRESH    = _mm256_set1_epi64x(HI_THRESH_D); //~sqrt(2)
    __m256d const ONE_F        = _mm256_set1_pd(ONE_F_D);
    __m256d const TWO          = _mm256_set1_pd(TWO_D);
    __m256d const ZERO         = _mm256_set1_pd(ZERO_D);

    __m256d const LN2_HI       = _mm256_set1_pd(LN2_HI_D);
    __m256d const LN2_LO       = _mm256_set1_pd(LN2_LO_D);
    //////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////// exp constants
    __m256d const L2E        = _mm256_set1_pd(L2E_D);
    __m256d const NEG_LN2_HI = _mm256_set1_pd(NEG_LN2_HI_D);
    __m256d const NEG_LN2_LO = _mm256_set1_pd(NEG_LN2_LO_D);

    __m256d const EXP_POLY_B = _mm256_set1_pd(EXP_POLY_B_D);
    __m256d const EXP_POLY_A = _mm256_set1_pd(EXP_POLY_A_D);
    __m256d const EXP_POLY_9 = _mm256_set1_pd(EXP_POLY_9_D);
    __m256d const EXP_POLY_8 = _mm256_set1_pd(EXP_POLY_8_D);
    __m256d const EXP_POLY_7 = _mm256_set1_pd(EXP_POLY_7_D);
    __m256d const EXP_POLY_6 = _mm256_set1_pd(EXP_POLY_6_D);
    __m256d const EXP_POLY_5 = _mm256_set1_pd(EXP_POLY_5_D);
    __m256d const EXP_POLY_4 = _mm256_set1_pd(EXP_POLY_4_D);
    __m256d const EXP_POLY_3 = _mm256_set1_pd(EXP_POLY_3_D);
    __m256d const EXP_POLY_2 = _mm256_set1_pd(EXP_POLY_2_D);
    __m256d const EXP_POLY_1 = _mm256_set1_pd(EXP_POLY_1_D);
    __m256d const EXP_POLY_0 = _mm256_set1_pd(EXP_POLY_0_D);

    __m256d const DBL2INT_CVT = _mm256_set1_pd(DBL2INT_CVT_D);

    __m256d const UPPERBOUND_1 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_1_D);
    __m256i const HI_ABS_MASK = _mm256_set1_epi64x(HI_ABS_MASK_D);

    //////////////////////////////////////////////////// pow constants
    __m256i const HI_MASK      = _mm256_set1_epi64x(HI_MASK_D);
    __m256d const SGN_EXP_MASK = (__m256d)_mm256_set1_epi64x(SGN_EXP_MASK_D);
    __m256i const ABS_MASK     = _mm256_set1_epi64x(ABS_MASK_D);
    __m256i const ONE         = _mm256_set1_epi64x(ONE_D);

    __m256i const TEN_23      = _mm256_set1_epi64x(TEN_23_D);
    __m256i const ALL_ONES_EXPONENT = _mm256_set1_epi64x(ALL_ONES_EXPONENT_D);


    __m256d abs_a, a_mut;
    __m256d_2 loga;
    __m256d t_hi, t_lo, tmp, e, bloga, prodx;

    __m256d_2 qq, cc, uu, tt;
    __m256d f, g, u, v, q, ulo, m;
    __m256i ihi, expo, expo_plus1;
    __m256d thresh_mask;
    __m256d b_is_one_mask;

    /*
     * Check whether all exponents in vector b are 1.0, and if so take
     * a quick exit.
     * Note: b_is_one_mask is needed later in case some elements in b are 1.0.
     */

    b_is_one_mask = _mm256_cmp_pd(b, ONE_F, _CMP_EQ_OQ);
    if (_mm256_movemask_pd(b_is_one_mask) == 0xf) {
        return a;
    }

    // *****************************************************************************************
    // computing log(abs(a))
    abs_a = _mm256_and_pd(a, (__m256d)ABS_MASK);
    a_mut = _mm256_and_pd(abs_a, HI_CONST_1);
    a_mut = _mm256_or_pd(a_mut, HI_CONST_2);
    m = (__m256d)_mm256_sub_epi32((__m256i)a_mut, HALFIFIER); // divide by 2 

    ihi = _mm256_and_si256((__m256i)a_mut, HI_MASK);
    thresh_mask = _mm256_cmp_pd((__m256d)ihi, (__m256d)HI_THRESH,_CMP_GT_OS); 
    m = _mm256_blendv_pd(a_mut, m, thresh_mask); 

    expo = _mm256_srli_epi64((__m256i)abs_a, D52_D);
    expo = _mm256_sub_epi64(expo, TEN_23);
    expo_plus1 = _mm256_add_epi64(expo, ONE); // add one to exponent instead
    expo = (__m256i)_mm256_blendv_pd((__m256d)expo, (__m256d)expo_plus1, thresh_mask);

    // begin computing log(m)
    f = _mm256_sub_pd(m, ONE_F);
    g = _mm256_add_pd(m, ONE_F);
    g = _mm256_div_pd(ONE_F, g); 

    u = _mm256_mul_pd(_mm256_mul_pd(TWO, f), g);

    // u = 2.0 * (m - 1.0) / (m + 1.0) 
    v = _mm256_mul_pd(u, u);

    // polynomial is used to approximate atanh(v)
    // an estrin evaluation scheme is used.
    __m256d c0 = _mm256_fmadd_pd(LOG_POLY_1, v, LOG_POLY_0);
    __m256d c2 = _mm256_fmadd_pd(LOG_POLY_3, v, LOG_POLY_2);
    __m256d c4 = _mm256_fmadd_pd(LOG_POLY_5, v, LOG_POLY_4);
    __m256d v2 = _mm256_mul_pd(v, v);
    __m256d v4 = _mm256_mul_pd(v2, v2);

    c0 = _mm256_fmadd_pd(c2, v2, c0);
    c4 = _mm256_fmadd_pd(LOG_POLY_6, v2, c4);
    q = _mm256_fmadd_pd(c4, v4, c0);
    q = _mm256_mul_pd(q, v);

    tmp = _mm256_mul_pd(TWO, _mm256_sub_pd(f, u));
    tmp = _mm256_fmadd_pd(neg(u), f, tmp);
    ulo = _mm256_mul_pd(g, tmp);

    // double-double computation begins
    qq.y = q;
    qq.x = ZERO;
    uu.y = u;
    uu.x = ulo;
    cc.y = CC_CONST_Y;
    cc.x = CC_CONST_X;

    qq = __internal_ddadd_yisdouble(cc, qq);

    // computing log(m) in double-double format
    cc.y = _mm256_mul_pd(uu.y, uu.y);
    cc.x = _mm256_fmsub_pd(uu.y, uu.y, cc.y);
    cc.x = _mm256_fmadd_pd(uu.y,
                           (__m256d)_mm256_add_epi32((__m256i)uu.x, HALFIFIER), 
                           cc.x); // u ** 2

    tt.y = _mm256_mul_pd(cc.y, uu.y);
    tt.x = _mm256_fmsub_pd(cc.y, uu.y, tt.y);
    tt.x = _mm256_fmadd_pd(cc.y, uu.x, tt.x);
    tt.x = _mm256_fmadd_pd(cc.x, uu.y, tt.x); // u ** 3 un-normalized

    uu = __internal_ddfma(qq, tt, uu);

    // computing log a = log(m) + log(2)*expo
    f = __internal_fast_int2dbl (expo);
    q = _mm256_fmadd_pd(f, LN2_HI, uu.y);
    tmp = _mm256_fmadd_pd(neg(f), LN2_HI, q);
    tmp = _mm256_sub_pd(tmp, uu.y);
    loga.y = q;
    loga.x = _mm256_sub_pd(uu.x, tmp);
    loga.x = _mm256_fmadd_pd(f, LN2_LO, loga.x);

    // finish log(a)
    // ******************************************************************************************
    
    // compute b * log(a)
    t_hi = _mm256_mul_pd(loga.y, b);
    t_lo = _mm256_fmsub_pd(loga.y, b, t_hi);
    t_lo = _mm256_fmadd_pd(loga.x, b, t_lo);
    bloga = e = _mm256_add_pd(t_hi, t_lo);
    prodx = _mm256_add_pd(_mm256_sub_pd(t_hi, e), t_lo);

    // ***************************************************************************************
    // computing exp(b * log(a))
    // calculating exponent; stored in the LO of each 64-bit block
    __m256i i = (__m256i) _mm256_fmadd_pd(bloga, L2E, DBL2INT_CVT);

    // calculate mantissa
    __m256d t = _mm256_sub_pd ((__m256d)i, DBL2INT_CVT);
    __m256d z = _mm256_fmadd_pd (t, NEG_LN2_HI, bloga);
    z = _mm256_fmadd_pd (t, NEG_LN2_LO, z);

    // use polynomial to calculate exp
    // mixed estrin/horner scheme: estrin on the higher 8 coefficients, horner on the lowest 4.
    // provided speedup without loss of precision compared to full horner
    __m256d t4 = _mm256_fmadd_pd(EXP_POLY_5, z, EXP_POLY_4);
    __m256d t6 = _mm256_fmadd_pd(EXP_POLY_7, z, EXP_POLY_6);
    __m256d t8 = _mm256_fmadd_pd(EXP_POLY_9, z, EXP_POLY_8);
    __m256d t10 = _mm256_fmadd_pd(EXP_POLY_B, z, EXP_POLY_A);

    __m256d z2 = _mm256_mul_pd(z, z);
    __m256d z4 = _mm256_mul_pd(z2, z2);

    t4 = _mm256_fmadd_pd(t6, z2, t4);
    t8 = _mm256_fmadd_pd(t10, z2, t8);
    t4 = _mm256_fmadd_pd(t8, z4, t4);
    
    t = _mm256_fmadd_pd(t4, z, EXP_POLY_3);
    t = _mm256_fmadd_pd(t, z, EXP_POLY_2);
    t = _mm256_fmadd_pd(t, z, EXP_POLY_1);
    t = _mm256_fmadd_pd(t, z, EXP_POLY_0); 

    // fast scale
    __m256i i_scale = _mm256_slli_epi64(i,D52_D); 
    __m256d ztemp = z = (__m256d)_mm256_add_epi32(i_scale, (__m256i)t);  

    // slowpath detection for exp
    __m256d abs_bloga = (__m256d)_mm256_and_si256((__m256i)bloga, HI_ABS_MASK); 
    int exp_slowmask = _mm256_movemask_pd(_mm256_cmp_pd(abs_bloga, UPPERBOUND_1, _CMP_GE_OS));

    z = _mm256_fmadd_pd(z, prodx, z);     

    if (__builtin_expect(exp_slowmask, 0)) {
        z = __pgm_exp_d_vec256_slowpath(i, t, bloga, ztemp, prodx); 
    }
    // finished exp(b * log (a))
    // ************************************************************************************************

    // Now special case if some elements of exponent (b) are 1.0.
    z = _mm256_blendv_pd(z, a, b_is_one_mask);

    // compute if we have special cases (inf, nan, etc). see man pow for full list of special cases
    __m256i detect_inf_nan = (__m256i)_mm256_add_pd(a, b);  // check for inf/nan
    __m256i overridemask = _mm256_cmpeq_epi64( (__m256i)a, (__m256i)ONE_F); //  if a == 1
    __m256i overridemask2 = _mm256_cmpeq_epi64(_mm256_and_si256(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);
    overridemask = _mm256_or_si256(overridemask, (__m256i)_mm256_cmp_pd(a, ZERO, _CMP_LE_OQ)); // if a < 0
         
    int specMask = _mm256_movemask_pd((__m256d)_mm256_or_si256(overridemask, overridemask2));
    if(__builtin_expect(specMask, 0)) {
        return __pgm_pow_d_vec256_special_cases(a, b, z);
    }
    return z;
}
     
