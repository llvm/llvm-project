
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
#include "dpow_defs.h"

extern "C" double __fsd_pow_fma3(double,double);


// struct representing a double-double number
// x, y represent the lo and hi parts respectively
struct __m128d_2 
{
    __m128d x, y;
};

// negates a number
inline
__m128d neg(__m128d a)
{
    __m128d const SGN_MASK = (__m128d)_mm_set1_epi64x(SGN_MASK_D);
    return _mm_xor_pd(a, SGN_MASK);
}

// double-double "fma"
inline
__m128d_2 __internal_ddfma (__m128d_2 x, __m128d_2 y, __m128d_2 z)
{
    __m128d e;
    __m128d_2 t, m;
    t.y = _mm_mul_sd(x.y, y.y);
    t.x = _mm_fmsub_sd(x.y, y.y, t.y);
    t.x = _mm_fmadd_sd(x.y, y.x, t.x);
    t.x = _mm_fmadd_sd(x.x, y.y, t.x);

    m.y = _mm_add_sd (z.y, t.y);
    e = _mm_sub_sd (z.y, m.y);
    m.x = _mm_add_sd (_mm_add_sd(_mm_add_sd (e, t.y), t.x), z.x);

    return m;
}

// special case of double-double addition, where |x| > |y| and y is a double.
inline
__m128d_2 __internal_ddadd_yisdouble(__m128d_2 x, __m128d_2 y)
{
    __m128d_2 z;
    __m128d e;
    z.y = _mm_add_sd (x.y, y.y);
    e = _mm_sub_sd (x.y, z.y);
    z.x = _mm_add_sd(_mm_add_sd (e, y.y), x.x);

    return z;
}

// casts int to double
inline
__m128d __internal_fast_int2dbl(__m128i a)
{
    __m128i const INT2DBL_HI = _mm_set1_epi64x(INT2DBL_HI_D);
    __m128i const INT2DBL_LO = _mm_set1_epi64x(INT2DBL_LO_D);
    __m128d const INT2DBL    = (__m128d)_mm_set1_epi64x(INT2DBL_D);

    __m128i t = _mm_xor_si128(INT2DBL_LO, a);
    t = _mm_blend_epi32(INT2DBL_HI, t, 0x5); 
    return _mm_sub_sd((__m128d)t, INT2DBL);
}

// slowpath for exp. used to improve accuracy on larger inputs
__m128d __attribute__ ((noinline)) __pgm_exp_d_scalar_slowpath(__m128i const i, __m128d const t, __m128d const bloga, __m128d z, __m128d prodx)
{
    __m128d const UPPERBOUND_1 = (__m128d)_mm_set1_epi64x(UPPERBOUND_1_D);
    __m128d const UPPERBOUND_2 = (__m128d)_mm_set1_epi64x(UPPERBOUND_2_D);
    __m128d const ZERO         = _mm_set1_pd(ZERO_D);
    __m128d const INF          = (__m128d)_mm_set1_epi64x(INF_D);
    __m128i const HI_ABS_MASK  = _mm_set1_epi64x(HI_ABS_MASK_D);
    __m128i const ABS_MASK     = _mm_set1_epi64x(ABS_MASK_D);
    __m128i const MULT_CONST   = _mm_set1_epi64x(MULT_CONST_D);

    __m128i abs_bloga = _mm_and_si128((__m128i)bloga, ABS_MASK);
    __m128d abs_lt = (__m128d)_mm_and_si128((__m128i)abs_bloga, HI_ABS_MASK);
    __m128d lt_zero_mask = _mm_cmp_sd(bloga, ZERO, _CMP_LT_OS); // compute bloga < 0.0  
    __m128d slowpath_mask = _mm_cmp_sd(abs_lt, UPPERBOUND_1, _CMP_LT_OS); 

    __m128d a_plus_inf = _mm_add_sd(bloga, INF); // check if bloga is too big      
    __m128d zero_inf_blend = _mm_blendv_pd(a_plus_inf, ZERO, lt_zero_mask);     

    __m128d accurate_scale_mask = (__m128d)_mm_cmp_sd(abs_lt, UPPERBOUND_2, _CMP_LT_OS); 

    // compute accurate scale
    __m128i k = _mm_srli_epi64(i, 1); // k = i / 2                             
    __m128i i_scale_acc = _mm_slli_epi64(k, D52_D);  // shift to HI and shift 20 
    k = _mm_sub_epi32(i, k);          // k = i - k                            
    __m128i i_scale_acc_2 = _mm_slli_epi64(k, D52_D);  // shift to HI and shift 20 
    __m128d multiplier = (__m128d)_mm_add_epi64(i_scale_acc_2, MULT_CONST);    
    multiplier = _mm_blendv_pd(ZERO, multiplier, accurate_scale_mask); // quick fix for overflows in case they're being trapped

    __m128d res = (__m128d)_mm_add_epi32(i_scale_acc, (__m128i)t);            
    res = _mm_mul_sd(res, multiplier);                                       

    __m128d slowpath_blend = _mm_blendv_pd(zero_inf_blend, res, accurate_scale_mask); 
    z = _mm_blendv_pd(slowpath_blend, z, slowpath_mask);

    __m128i isinf_mask = _mm_cmpeq_epi64((__m128i)INF, (__m128i)z); // special case for inf
    __m128d z_fixed = _mm_fmadd_sd(z, prodx, z); // add only if not inf
    z = _mm_blendv_pd(z_fixed, z, (__m128d)isinf_mask);

    return z;
}

// special cases for pow
// implementation derived from cudart/device_functions_impl.c
__m128d __attribute__ ((noinline)) __pgm_pow_d_scalar_special_cases(__m128d const a, __m128d const b, __m128d t)
{
   __m128i const HI_MASK       = _mm_set1_epi64x(HI_MASK_D);
   __m128d const SGN_EXP_MASK  = (__m128d)_mm_set1_epi64x(SGN_EXP_MASK_D);
   __m128i const ABS_MASK      = _mm_set1_epi64x(ABS_MASK_D);
   __m128d const NEG_ONE       = _mm_set1_pd(NEG_ONE_D);
   __m128d const ZERO          = _mm_set1_pd(ZERO_D);
   __m128d const ONE           = _mm_set1_pd(ONE_D);
   __m128d const HALF          = _mm_set1_pd(HALF_D);
   __m128d const SGN_MASK      = (__m128d)_mm_set1_epi64x(SGN_MASK_D);
   __m128d const INF           = (__m128d)_mm_set1_epi64x(INF_D);
   __m128i const INF_FAKE      = _mm_set1_epi64x(INF_FAKE_D);
   __m128d const NAN_MASK      = (__m128d)_mm_set1_epi64x(NAN_MASK_D);
   __m128d const NEG_ONE_CONST = (__m128d)_mm_set1_epi64x(NEG_ONE_CONST_D);

   __m128i const TEN_23        = _mm_set1_epi64x(TEN_23_D);
   __m128i const ELEVEN        = _mm_set1_epi64x(ELEVEN_D);

    __m128i a_sign, b_sign, bIsOddInteger;
    __m128d abs_a, abs_b;
    __m128i shiftb;

    a_sign = (__m128i)_mm_and_pd(a, SGN_MASK);
    b_sign = (__m128i)_mm_and_pd(b, SGN_MASK);
    abs_a = _mm_and_pd(a, (__m128d)ABS_MASK); 
    abs_b = _mm_and_pd(b, (__m128d)ABS_MASK);

    // determining if b is an odd integer, since there are special cases for it
    shiftb = (__m128i)_mm_and_pd (b, SGN_EXP_MASK);
    shiftb = _mm_srli_epi64(shiftb, D52_D);
    shiftb = _mm_sub_epi64(shiftb, TEN_23);
    shiftb = _mm_add_epi64(shiftb, ELEVEN);

    __m128i b_is_half = _mm_cmpeq_epi64((__m128i)abs_b, (__m128i)HALF);
    bIsOddInteger = _mm_sllv_epi64((__m128i)b, shiftb);
    bIsOddInteger = _mm_cmpeq_epi64((__m128i)SGN_MASK, bIsOddInteger);
    // fix for b = +/-0.5 being incorrectly identified as an odd integer
    bIsOddInteger = _mm_andnot_si128(b_is_half, bIsOddInteger);

    // corner cases where a <= 0
    // if ((ahi < 0) && bIsOddInteger)
    __m128d ahi_lt_0 = (__m128d)_mm_cmpeq_epi64(a_sign, (__m128i)ZERO);
    __m128d ahilt0_and_boddint_mask = _mm_andnot_pd(ahi_lt_0, (__m128d)bIsOddInteger);

    t = _mm_blendv_pd(t, neg(t), ahilt0_and_boddint_mask);

    // else if ((ahi < 0) && (b != trunc(b)))
    __m128d b_ne_trunc = _mm_cmp_sd(b, _mm_round_pd(b, _MM_FROUND_TO_ZERO), _CMP_NEQ_UQ);
    __m128d nan_mask = _mm_andnot_pd(ahi_lt_0, b_ne_trunc);
    t = _mm_blendv_pd(t, NAN_MASK, nan_mask);

    // if (a == 0.0)
    __m128d a_is_0_mask = (__m128d)_mm_cmpeq_epi64((__m128i)abs_a, (__m128i)ZERO);
    __m128d thi_when_ais0 = ZERO;

    // if (bIsOddInteger && a == 0.0)
    thi_when_ais0 = _mm_blendv_pd(thi_when_ais0, (__m128d)a, (__m128d)bIsOddInteger);

    // if (bhi < 0 && a == 0.0)
    __m128d bhi_lt_0 = (__m128d)_mm_cmpeq_epi64(b_sign, (__m128i)ZERO); //this mask is inverted
    __m128d thi_or_INF = _mm_or_pd(thi_when_ais0, INF);
    thi_when_ais0 = _mm_blendv_pd(thi_or_INF, thi_when_ais0, bhi_lt_0);
    __m128d t_when_ais0 = _mm_and_pd(thi_when_ais0, (__m128d)HI_MASK);

    t = _mm_blendv_pd(t, t_when_ais0, a_is_0_mask);

    // else if (a is INF)
    __m128d a_inf_mask = (__m128d)_mm_cmpeq_epi64((__m128i)abs_a, (__m128i)INF);
    // use bhi_lt_0 backwards to evaluate bhi >= 0
    __m128d thi_when_aisinf = _mm_blendv_pd(thi_when_aisinf, INF, bhi_lt_0);

    // now evaluate ((ahi < 0) && bIsOddInteger)
    __m128d thi_xor_sgn = _mm_xor_pd(thi_when_aisinf, SGN_MASK);
    thi_when_aisinf = _mm_blendv_pd(thi_when_aisinf, thi_xor_sgn, ahilt0_and_boddint_mask);

    t = _mm_blendv_pd(t, thi_when_aisinf, a_inf_mask);

    // else if (abs(b) is INF)
    __m128d b_inf_mask = (__m128d)_mm_cmpeq_epi64((__m128i)abs_b, (__m128i)INF);
    __m128d thi_when_bisinf = ZERO;
    __m128d absa_gt_one = _mm_cmp_sd(abs_a, ONE, _CMP_GT_OS); // evaluating (abs(a) > 1)
    thi_when_bisinf = _mm_blendv_pd(thi_when_bisinf, INF, absa_gt_one);

    __m128d thi_xor_inf = _mm_xor_pd(thi_when_bisinf, INF);
    thi_when_bisinf = _mm_blendv_pd(thi_xor_inf, thi_when_bisinf, bhi_lt_0); // bhi < 0

    __m128d a_is_negone = (__m128d)_mm_cmpeq_epi64((__m128i)a, (__m128i)NEG_ONE);
    thi_when_bisinf = _mm_blendv_pd(thi_when_bisinf, NEG_ONE_CONST, a_is_negone); //a == -1

    t = _mm_blendv_pd(t, thi_when_bisinf, b_inf_mask);

    // if(a is NAN || B is NAN) <=> !(a is a number && b is a number)
    __m128i a_nan_mask = _mm_cmpeq_epi64(_mm_max_epu32((__m128i)abs_a, INF_FAKE), INF_FAKE);
    __m128i b_nan_mask = _mm_cmpeq_epi64(_mm_max_epu32((__m128i)abs_b, INF_FAKE), INF_FAKE);
    __m128i aorb_nan_mask = _mm_and_si128(a_nan_mask, b_nan_mask); //this mask is inverted
    t = _mm_blendv_pd(_mm_add_sd(a,b), t, (__m128d)aorb_nan_mask);

    // if a == 1 or b == 0, answer = 1 
    __m128d a_is_one_mask = (__m128d)_mm_cmpeq_epi64((__m128i)a, (__m128i)ONE);
    __m128d b_is_zero_mask = (__m128d)_mm_cmpeq_epi64((__m128i)abs_b, (__m128i)ZERO);
    __m128d ans_equals_one_mask = _mm_or_pd(a_is_one_mask, b_is_zero_mask);
    t = _mm_blendv_pd(t, ONE, ans_equals_one_mask);
    // ****************************************************************************************** 
//  
//
   __m128i a_is_neg = (__m128i)_mm_cmp_pd(a, ZERO, _CMP_LT_OS);
   int a_is_neg_flag = _mm_movemask_epi8((__m128i)a_is_neg);

   __m128i b_is_integer = (__m128i)_mm_cmp_pd(b, _mm_floor_pd(b), _CMP_EQ_OQ);
   int b_is_integer_flag = _mm_movemask_epi8((__m128i)b_is_integer);
 
   __m128i b_is_lt_zero = (__m128i)_mm_cmp_pd(b, ZERO, _CMP_LT_OS);
   int b_is_lt_zero_flag = _mm_movemask_epi8((__m128i)b_is_lt_zero);

   __m128d const MINUS_ZERO = _mm_set1_pd((double)-0.0);
   __m128i a_is_pos_zero = _mm_cmpeq_epi64( (__m128i)a, (__m128i)ZERO);
   __m128i a_is_neg_zero = _mm_cmpeq_epi64( (__m128i)a, (__m128i)MINUS_ZERO);
   __m128i a_is_any_zero = _mm_or_si128(a_is_pos_zero, a_is_neg_zero);
   int a_is_any_zero_flag = _mm_movemask_epi8((__m128i)a_is_any_zero);

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
         __m128d volatile invop = _mm_sqrt_pd(a);
      }

      if (a_is_any_zero_flag && b_is_lt_zero_flag) {
         __m128d volatile divXzero = _mm_div_pd(ONE,ZERO);
      }
   
    return t;
}

double __fsd_pow_fma3(double const a_in, double const b_in)
{
    //////////////////////////////////////////////////// log constants
    __m128d const LOG_POLY_6 = _mm_set1_pd(LOG_POLY_6_D);
    __m128d const LOG_POLY_5 = _mm_set1_pd(LOG_POLY_5_D);
    __m128d const LOG_POLY_4 = _mm_set1_pd(LOG_POLY_4_D);
    __m128d const LOG_POLY_3 = _mm_set1_pd(LOG_POLY_3_D);
    __m128d const LOG_POLY_2 = _mm_set1_pd(LOG_POLY_2_D);
    __m128d const LOG_POLY_1 = _mm_set1_pd(LOG_POLY_1_D);
    __m128d const LOG_POLY_0 = _mm_set1_pd(LOG_POLY_0_D);

    __m128d const CC_CONST_Y   = _mm_set1_pd(CC_CONST_Y_D);
    __m128d const CC_CONST_X   = _mm_set1_pd(CC_CONST_X_D);

    __m128i const EXPO_MASK    = _mm_set1_epi64x(EXPO_MASK_D);
    __m128d const HI_CONST_1   = (__m128d)_mm_set1_epi64x(HI_CONST_1_D);
    __m128d const HI_CONST_2   = (__m128d)_mm_set1_epi64x(HI_CONST_2_D);
    __m128i const HALFIFIER    = _mm_set1_epi64x(HALFIFIER_D); 
    __m128i const HI_THRESH    = _mm_set1_epi64x(HI_THRESH_D); //~sqrt(2)
    __m128d const ONE_F        = _mm_set1_pd(ONE_F_D);
    __m128d const TWO          = _mm_set1_pd(TWO_D);
    __m128d const ZERO         = _mm_set1_pd(ZERO_D);

    __m128d const LN2_HI       = _mm_set1_pd(LN2_HI_D);
    __m128d const LN2_LO       = _mm_set1_pd(LN2_LO_D);
    //////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////// exp constants
    __m128d const L2E        = _mm_set1_pd(L2E_D);
    __m128d const NEG_LN2_HI = _mm_set1_pd(NEG_LN2_HI_D);
    __m128d const NEG_LN2_LO = _mm_set1_pd(NEG_LN2_LO_D);

    __m128d const EXP_POLY_B = _mm_set1_pd(EXP_POLY_B_D);
    __m128d const EXP_POLY_A = _mm_set1_pd(EXP_POLY_A_D);
    __m128d const EXP_POLY_9 = _mm_set1_pd(EXP_POLY_9_D);
    __m128d const EXP_POLY_8 = _mm_set1_pd(EXP_POLY_8_D);
    __m128d const EXP_POLY_7 = _mm_set1_pd(EXP_POLY_7_D);
    __m128d const EXP_POLY_6 = _mm_set1_pd(EXP_POLY_6_D);
    __m128d const EXP_POLY_5 = _mm_set1_pd(EXP_POLY_5_D);
    __m128d const EXP_POLY_4 = _mm_set1_pd(EXP_POLY_4_D);
    __m128d const EXP_POLY_3 = _mm_set1_pd(EXP_POLY_3_D);
    __m128d const EXP_POLY_2 = _mm_set1_pd(EXP_POLY_2_D);
    __m128d const EXP_POLY_1 = _mm_set1_pd(EXP_POLY_1_D);
    __m128d const EXP_POLY_0 = _mm_set1_pd(EXP_POLY_0_D);

    __m128d const DBL2INT_CVT = _mm_set1_pd(DBL2INT_CVT_D);

    __m128d const UPPERBOUND_1 = (__m128d)_mm_set1_epi64x(UPPERBOUND_1_D);
    __m128i const HI_ABS_MASK = _mm_set1_epi64x(HI_ABS_MASK_D);

    //////////////////////////////////////////////////// pow constants
    __m128i const HI_MASK      = _mm_set1_epi64x(HI_MASK_D);
    __m128d const SGN_EXP_MASK = (__m128d)_mm_set1_epi64x(SGN_EXP_MASK_D);
    __m128i const ABS_MASK     = _mm_set1_epi64x(ABS_MASK_D);
    __m128i const ONE         = _mm_set1_epi64x(ONE_D);

    __m128i const TEN_23      = _mm_set1_epi64x(TEN_23_D);
    __m128i const ALL_ONES_EXPONENT = _mm_set1_epi64x(ALL_ONES_EXPONENT_D);

    __m128d abs_a, a_mut;
    __m128d_2 loga;
    __m128d t_hi, t_lo, tmp, e, bloga, prodx;

    __m128d_2 qq, cc, uu, tt;
    __m128d f, g, u, v, q, ulo, m;
    __m128i ihi, expo, expo_plus1;
    __m128d thresh_mask;

    __m128d a = _mm_set1_pd(a_in);
    __m128d b = _mm_set1_pd(b_in);


    /*
     * Check for exponent(b) being 1.0 and take a quick exit.
     */
    if (b_in == 1.0) {
        return a_in;
    }
    
    // *****************************************************************************************
    // computing log(abs(a))
    abs_a = _mm_and_pd(a, (__m128d)ABS_MASK);
    a_mut = _mm_and_pd(abs_a, HI_CONST_1);
    a_mut = _mm_or_pd(a_mut, HI_CONST_2);
    m = (__m128d)_mm_sub_epi32((__m128i)a_mut, HALFIFIER); // divide by 2 

    ihi = _mm_and_si128((__m128i)a_mut, HI_MASK);
    thresh_mask = _mm_cmp_sd((__m128d)ihi, (__m128d)HI_THRESH,_CMP_GT_OS); 
    m = _mm_blendv_pd(a_mut, m, thresh_mask); 

    expo = _mm_srli_epi64((__m128i)abs_a, D52_D);
    expo = _mm_sub_epi64(expo, TEN_23);
    expo_plus1 = _mm_add_epi64(expo, ONE); // add one to exponent instead
    expo = (__m128i)_mm_blendv_pd((__m128d)expo, (__m128d)expo_plus1, thresh_mask);

    // begin computing log(m)
    f = _mm_sub_sd(m, ONE_F);
    g = _mm_add_sd(m, ONE_F);
    g = _mm_div_sd(ONE_F, g); 

    u = _mm_mul_sd(_mm_mul_sd(TWO, f), g);

    // u = 2.0 * (m - 1.0) / (m + 1.0) 
    v = _mm_mul_sd(u, u);

    // polynomial is used to approximate atanh(v)
    // an estrin evaluation scheme is used.
    __m128d c0 = _mm_fmadd_sd(LOG_POLY_1, v, LOG_POLY_0);
    __m128d c2 = _mm_fmadd_sd(LOG_POLY_3, v, LOG_POLY_2);
    __m128d c4 = _mm_fmadd_sd(LOG_POLY_5, v, LOG_POLY_4);
    __m128d v2 = _mm_mul_sd(v, v);
    __m128d v4 = _mm_mul_sd(v2, v2);

    c0 = _mm_fmadd_sd(c2, v2, c0);
    c4 = _mm_fmadd_sd(LOG_POLY_6, v2, c4);
    q = _mm_fmadd_sd(c4, v4, c0);
    q = _mm_mul_sd(q, v);

    tmp = _mm_mul_sd(TWO, _mm_sub_sd(f, u));
    tmp = _mm_fmadd_sd(neg(u), f, tmp);
    ulo = _mm_mul_sd(g, tmp);

    // double-double computation begins
    qq.y = q;
    qq.x = ZERO;
    uu.y = u;
    uu.x = ulo;
    cc.y = CC_CONST_Y;
    cc.x = CC_CONST_X;

    qq = __internal_ddadd_yisdouble(cc, qq);

    // computing log(m) in double-double format
    cc.y = _mm_mul_sd(uu.y, uu.y);
    cc.x = _mm_fmsub_sd(uu.y, uu.y, cc.y);
    cc.x = _mm_fmadd_sd(uu.y,
                           (__m128d)_mm_add_epi32((__m128i)uu.x, HALFIFIER), 
                           cc.x); // u ** 2

    tt.y = _mm_mul_sd(cc.y, uu.y);
    tt.x = _mm_fmsub_sd(cc.y, uu.y, tt.y);
    tt.x = _mm_fmadd_sd(cc.y, uu.x, tt.x);
    tt.x = _mm_fmadd_sd(cc.x, uu.y, tt.x); // u ** 3 un-normalized

    uu = __internal_ddfma(qq, tt, uu);

    // computing log a = log(m) + log(2)*expo
    f = __internal_fast_int2dbl (expo);
    q = _mm_fmadd_sd(f, LN2_HI, uu.y);
    tmp = _mm_fmadd_sd(neg(f), LN2_HI, q);
    tmp = _mm_sub_sd(tmp, uu.y);
    loga.y = q;
    loga.x = _mm_sub_sd(uu.x, tmp);
    loga.x = _mm_fmadd_sd(f, LN2_LO, loga.x);

    // finish log(a)
    // ******************************************************************************************
    
    // compute b * log(a)
    t_hi = _mm_mul_sd(loga.y, b);
    t_lo = _mm_fmsub_sd(loga.y, b, t_hi);
    t_lo = _mm_fmadd_sd(loga.x, b, t_lo);
    bloga = e = _mm_add_sd(t_hi, t_lo);
    prodx = _mm_add_sd(_mm_sub_sd(t_hi, e), t_lo);

    // ***************************************************************************************
    // computing exp(b * log(a))
    // calculating exponent; stored in the LO of each 64-bit block
    __m128i i = (__m128i) _mm_fmadd_sd(bloga, L2E, DBL2INT_CVT);

    // calculate mantissa
    __m128d t = _mm_sub_sd ((__m128d)i, DBL2INT_CVT);
    __m128d z = _mm_fmadd_sd (t, NEG_LN2_HI, bloga);
    z = _mm_fmadd_sd (t, NEG_LN2_LO, z);

    // use polynomial to calculate exp
    // mixed estrin/horner scheme: estrin on the higher 8 coefficients, horner on the lowest 4.
    // provided speedup without loss of precision compared to full horner
    __m128d t4 = _mm_fmadd_sd(EXP_POLY_5, z, EXP_POLY_4);
    __m128d t6 = _mm_fmadd_sd(EXP_POLY_7, z, EXP_POLY_6);
    __m128d t8 = _mm_fmadd_sd(EXP_POLY_9, z, EXP_POLY_8);
    __m128d t10 = _mm_fmadd_sd(EXP_POLY_B, z, EXP_POLY_A);

    __m128d z2 = _mm_mul_sd(z, z);
    __m128d z4 = _mm_mul_sd(z2, z2);

    t4 = _mm_fmadd_sd(t6, z2, t4);
    t8 = _mm_fmadd_sd(t10, z2, t8);
    t4 = _mm_fmadd_sd(t8, z4, t4);
    
    t = _mm_fmadd_sd(t4, z, EXP_POLY_3);
    t = _mm_fmadd_sd(t, z, EXP_POLY_2);
    t = _mm_fmadd_sd(t, z, EXP_POLY_1);
    t = _mm_fmadd_sd(t, z, EXP_POLY_0); 

    // fast scale
    __m128i i_scale = _mm_slli_epi64(i,D52_D); 
    __m128d ztemp = z = (__m128d)_mm_add_epi32(i_scale, (__m128i)t);  

    // slowpath detection for exp
    __m128d abs_bloga = (__m128d)_mm_and_si128((__m128i)bloga, HI_ABS_MASK); 
#if defined(TARGET_LINUX_POWER)
    int exp_slowmask = _vec_any_nz((__m128i)_mm_cmp_sd(abs_bloga, UPPERBOUND_1, _CMP_GE_OS));
#else
    int exp_slowmask = _mm_movemask_pd(_mm_cmp_sd(abs_bloga, UPPERBOUND_1, _CMP_GE_OS));
#endif

    z = _mm_fmadd_sd(z, prodx, z);     

    if (__builtin_expect(exp_slowmask, 0)) {
        z = __pgm_exp_d_scalar_slowpath(i, t, bloga, ztemp, prodx); 
    }
    // finished exp(b * log (a))
    // ************************************************************************************************

    // compute if we have special cases (inf, nan, etc). see man pow for full list of special cases
    __m128i detect_inf_nan = (__m128i)_mm_add_sd(a, b);  // check for inf/nan
    __m128i overridemask = _mm_cmpeq_epi64( (__m128i)a, (__m128i)ONE_F); //  if a == 1
    __m128i overridemask2 = _mm_cmpeq_epi64(_mm_and_si128(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);
    overridemask = _mm_or_si128(overridemask, (__m128i)_mm_cmp_sd(a, ZERO, _CMP_LE_OQ)); // if a < 0
         
#if defined(TARGET_LINUX_POWER)
    int specMask = _vec_any_nz((__m128i)_mm_or_si128(overridemask, overridemask2));
#else
    int specMask = _mm_movemask_pd((__m128d)_mm_or_si128(overridemask, overridemask2));
#endif
    if(__builtin_expect(specMask, 0)) {
        return _mm_cvtsd_f64(__pgm_pow_d_scalar_special_cases(a, b, z));
    }
    return _mm_cvtsd_f64(z);
}
     
