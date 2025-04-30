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
#include "dexp_defs.h"

extern "C" __m256d __fvd_exp_fma3_256(__m256d);


// handles large cases as well as special cases such as infinities and NaNs
__m256d __pgm_exp_d_vec256_slowpath(__m256d const a, __m256i const i, __m256d const t,  __m256d const z)
{
    __m256d const INF        = (__m256d)_mm256_set1_epi64x(INF_D);
    __m256d const ZERO       = _mm256_set1_pd(ZERO_D);
    __m256i const HI_ABS_MASK = _mm256_set1_epi64x(HI_ABS_MASK_D);
    __m256d const UPPERBOUND_1 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_1_D);
    __m256d const UPPERBOUND_2 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_2_D);
    __m256i const MULT_CONST = _mm256_set1_epi64x(MULT_CONST_D);

    __m256d abs_lt = (__m256d)_mm256_and_si256((__m256i)a, HI_ABS_MASK);                    

    __m256d slowpath_mask = (__m256d)_mm256_cmp_pd(abs_lt, UPPERBOUND_1, 1);       
    __m256d lt_zero_mask = _mm256_cmp_pd(a, ZERO, 1); // compute a < 0.0           

    __m256d a_plus_inf = _mm256_add_pd(a, INF); // check if a is too big           
    __m256d zero_inf_blend = _mm256_blendv_pd(a_plus_inf, ZERO, lt_zero_mask);     

    __m256d accurate_scale_mask = (__m256d)_mm256_cmp_pd(abs_lt, UPPERBOUND_2, 1); 

    // compute accurate scale
    __m256i k = _mm256_srli_epi64(i, 1); // k = i / 2                              
    __m256i i_scale_acc = _mm256_slli_epi64(k, SCALE_D);  // shift to HI and shift 20   

    k = _mm256_sub_epi32(i, k);          // k = i - k                              
    __m256i i_scale_acc_2 = _mm256_slli_epi64(k, SCALE_D);  // shift to HI and shift 20 
    __m256d multiplier = (__m256d)_mm256_add_epi64(i_scale_acc_2, MULT_CONST);     

    __m256d res = (__m256d)_mm256_add_epi32(i_scale_acc, (__m256i)t);              
    res = _mm256_mul_pd(res, multiplier);                                          

    __m256d slowpath_blend = _mm256_blendv_pd(zero_inf_blend, res, accurate_scale_mask); 
    return  _mm256_blendv_pd(slowpath_blend, z, slowpath_mask);
}


__m256d __fvd_exp_fma3_256(__m256d const a)
{
    __m256d const L2E        = _mm256_set1_pd(L2E_D);
    __m256d const NEG_LN2_HI = _mm256_set1_pd(NEG_LN2_HI_D);
    __m256d const NEG_LN2_LO = _mm256_set1_pd(NEG_LN2_LO_D);
    __m256d const ZERO       = _mm256_set1_pd(ZERO_D);
    __m256d const INF        = (__m256d)_mm256_set1_epi64x(INF_D);

    __m256d const EXP_POLY_11 = _mm256_set1_pd(EXP_POLY_11_D);
    __m256d const EXP_POLY_10 = _mm256_set1_pd(EXP_POLY_10_D);
    __m256d const EXP_POLY_9  = _mm256_set1_pd(EXP_POLY_9_D);
    __m256d const EXP_POLY_8  = _mm256_set1_pd(EXP_POLY_8_D);
    __m256d const EXP_POLY_7  = _mm256_set1_pd(EXP_POLY_7_D);
    __m256d const EXP_POLY_6  = _mm256_set1_pd(EXP_POLY_6_D);
    __m256d const EXP_POLY_5  = _mm256_set1_pd(EXP_POLY_5_D);
    __m256d const EXP_POLY_4  = _mm256_set1_pd(EXP_POLY_4_D);
    __m256d const EXP_POLY_3  = _mm256_set1_pd(EXP_POLY_3_D);
    __m256d const EXP_POLY_2  = _mm256_set1_pd(EXP_POLY_2_D);
    __m256d const EXP_POLY_1  = _mm256_set1_pd(EXP_POLY_1_D);
    __m256d const EXP_POLY_0  = _mm256_set1_pd(EXP_POLY_0_D);

    __m256d const DBL2INT_CVT = _mm256_set1_pd(DBL2INT_CVT_D);
    __m256d const UPPERBOUND_1 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_1_D);
    __m256d const UPPERBOUND_2 = (__m256d)_mm256_set1_epi64x(UPPERBOUND_2_D);

    __m256i const MULT_CONST = _mm256_set1_epi64x(MULT_CONST_D);
    __m256i const HI_ABS_MASK = _mm256_set1_epi64x(HI_ABS_MASK_D);

    // calculating exponent; stored in the LO of each 64-bit block
    __m256i i = (__m256i) _mm256_fmadd_pd(a, L2E, DBL2INT_CVT);

    // calculate mantissa
    //fast mul rint
    __m256d t = _mm256_sub_pd (_mm256_fmadd_pd(a, L2E, DBL2INT_CVT), DBL2INT_CVT);
    __m256d m = _mm256_fmadd_pd (t, NEG_LN2_HI, a);
    m = _mm256_fmadd_pd (t, NEG_LN2_LO, m);

    // evaluate highest 8 terms of polynomial with estrin, then switch to horner
    __m256d z10 = _mm256_fmadd_pd(EXP_POLY_11, m, EXP_POLY_10);
    __m256d z8  = _mm256_fmadd_pd(EXP_POLY_9, m, EXP_POLY_8);
    __m256d z6  = _mm256_fmadd_pd(EXP_POLY_7, m, EXP_POLY_6);
    __m256d z4  = _mm256_fmadd_pd(EXP_POLY_5, m, EXP_POLY_4);

    __m256d m2 = _mm256_mul_pd(m, m);
    z8 = _mm256_fmadd_pd(z10, m2, z8);
    z4 = _mm256_fmadd_pd(z6, m2, z4); 
    
    __m256d m4 = _mm256_mul_pd(m2, m2);
    z4 = _mm256_fmadd_pd(z8, m4, z4);

    t = _mm256_fmadd_pd(z4, m, EXP_POLY_3);
    t = _mm256_fmadd_pd(t, m, EXP_POLY_2);
    t = _mm256_fmadd_pd(t, m, EXP_POLY_1);
    t = _mm256_fmadd_pd(t, m, EXP_POLY_0);
    
    // fast scale
    __m256i i_scale = _mm256_slli_epi64(i, SCALE_D); 
    __m256d z = (__m256d)_mm256_add_epi32(i_scale, (__m256i)t); 

    __m256d abs_a = (__m256d)_mm256_and_si256((__m256i)a, HI_ABS_MASK);

    int exp_slowmask = _mm256_movemask_epi8(_mm256_cmpgt_epi64((__m256i)abs_a, (__m256i)UPPERBOUND_1));

//    if (exp_slowmask) {
//        return __pgm_exp_d_vec256_slowpath(a, i, t, z);
//    }


//    int exp_slowmask = _mm256_movemask_pd(_mm256_cmp_pd(abs_a, UPPERBOUND_1, _CMP_GE_OS));
    
    if (__builtin_expect(exp_slowmask, 0)) {
        return __pgm_exp_d_vec256_slowpath(a, i, t, z);
    }

    return z;
} 
