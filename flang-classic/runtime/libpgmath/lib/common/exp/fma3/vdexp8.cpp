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
#include "dexp_defs.h"

/*
 * Using the new mask registers {k}, AVX512 offers opportunites to improve 
 * comparing and merging of registers compared to the AVX/AVX2 instruction
 * sets.
 *
 * However for purposes of symmetry between the 128 and 256 implementations
 * of EXP(), try to keep source files similar.
 *
 */

extern "C" __m512d FCN_AVX512(__fvd_exp_fma3)(__m512d);

// handles large cases as well as special cases such as infinities and NaNs
static __m512d __pgm_exp_d_vec512_slowpath(__m512d const a, __m512i const i, __m512d const t,  __m512d const z)
{
    __m512d const INF        = (__m512d)_mm512_set1_epi64(INF_D);
    __m512d const ZERO       = _mm512_set1_pd(ZERO_D);
    __m512i const HI_ABS_MASK = _mm512_set1_epi64(HI_ABS_MASK_D);
    __m512d const UPPERBOUND_1 = (__m512d)_mm512_set1_epi64(UPPERBOUND_1_D);
    __m512d const UPPERBOUND_2 = (__m512d)_mm512_set1_epi64(UPPERBOUND_2_D);
    __m512i const MULT_CONST = _mm512_set1_epi64(MULT_CONST_D);

    __m512d abs_lt = (__m512d)_mm512_and_si512((__m512i)a, HI_ABS_MASK);                    

    __m512d slowpath_mask = (__m512d)_MM512_CMP_PD(abs_lt, UPPERBOUND_1, 1);       
    __m512d lt_zero_mask = _MM512_CMP_PD(a, ZERO, 1); // compute a < 0.0           

    __m512d a_plus_inf = _mm512_add_pd(a, INF); // check if a is too big           
    __m512d zero_inf_blend = _MM512_BLENDV_PD(a_plus_inf, ZERO, lt_zero_mask);     

    __m512d accurate_scale_mask = (__m512d)_MM512_CMP_PD(abs_lt, UPPERBOUND_2, 1); 

    // compute accurate scale
    __m512i k = _mm512_srli_epi64(i, 1); // k = i / 2                              
    __m512i i_scale_acc = _mm512_slli_epi64(k, SCALE_D);  // shift to HI and shift 20   

    k = _mm512_sub_epi32(i, k);          // k = i - k                              
    __m512i i_scale_acc_2 = _mm512_slli_epi64(k, SCALE_D);  // shift to HI and shift 20 
    __m512d multiplier = (__m512d)_mm512_add_epi64(i_scale_acc_2, MULT_CONST);     

    __m512d res = (__m512d)_mm512_add_epi32(i_scale_acc, (__m512i)t);              
    res = _mm512_mul_pd(res, multiplier);                                          

    __m512d slowpath_blend = _MM512_BLENDV_PD(zero_inf_blend, res, accurate_scale_mask); 
    return  _MM512_BLENDV_PD(slowpath_blend, z, slowpath_mask);
}


__m512d FCN_AVX512(__fvd_exp_fma3)(__m512d const a)
{
    __m512d const L2E        = _mm512_set1_pd(L2E_D);
    __m512d const NEG_LN2_HI = _mm512_set1_pd(NEG_LN2_HI_D);
    __m512d const NEG_LN2_LO = _mm512_set1_pd(NEG_LN2_LO_D);
    __m512d const ZERO       = _mm512_set1_pd(ZERO_D);
    __m512d const INF        = (__m512d)_mm512_set1_epi64(INF_D);

    __m512d const EXP_POLY_11 = _mm512_set1_pd(EXP_POLY_11_D);
    __m512d const EXP_POLY_10 = _mm512_set1_pd(EXP_POLY_10_D);
    __m512d const EXP_POLY_9  = _mm512_set1_pd(EXP_POLY_9_D);
    __m512d const EXP_POLY_8  = _mm512_set1_pd(EXP_POLY_8_D);
    __m512d const EXP_POLY_7  = _mm512_set1_pd(EXP_POLY_7_D);
    __m512d const EXP_POLY_6  = _mm512_set1_pd(EXP_POLY_6_D);
    __m512d const EXP_POLY_5  = _mm512_set1_pd(EXP_POLY_5_D);
    __m512d const EXP_POLY_4  = _mm512_set1_pd(EXP_POLY_4_D);
    __m512d const EXP_POLY_3  = _mm512_set1_pd(EXP_POLY_3_D);
    __m512d const EXP_POLY_2  = _mm512_set1_pd(EXP_POLY_2_D);
    __m512d const EXP_POLY_1  = _mm512_set1_pd(EXP_POLY_1_D);
    __m512d const EXP_POLY_0  = _mm512_set1_pd(EXP_POLY_0_D);

    __m512d const DBL2INT_CVT = _mm512_set1_pd(DBL2INT_CVT_D);
    __m512d const UPPERBOUND_1 = (__m512d)_mm512_set1_epi64(UPPERBOUND_1_D);
    __m512d const UPPERBOUND_2 = (__m512d)_mm512_set1_epi64(UPPERBOUND_2_D);

    __m512i const MULT_CONST = _mm512_set1_epi64(MULT_CONST_D);
    __m512i const HI_ABS_MASK = _mm512_set1_epi64(HI_ABS_MASK_D);

    // calculating exponent; stored in the LO of each 64-bit block
    __m512i i = (__m512i) _mm512_fmadd_pd(a, L2E, DBL2INT_CVT);

    // calculate mantissa
    //fast mul rint
    __m512d t = _mm512_sub_pd (_mm512_fmadd_pd(a, L2E, DBL2INT_CVT), DBL2INT_CVT);
    __m512d m = _mm512_fmadd_pd (t, NEG_LN2_HI, a);
    m = _mm512_fmadd_pd (t, NEG_LN2_LO, m);

    // evaluate highest 8 terms of polynomial with estrin, then switch to horner
    __m512d z10 = _mm512_fmadd_pd(EXP_POLY_11, m, EXP_POLY_10);
    __m512d z8  = _mm512_fmadd_pd(EXP_POLY_9, m, EXP_POLY_8);
    __m512d z6  = _mm512_fmadd_pd(EXP_POLY_7, m, EXP_POLY_6);
    __m512d z4  = _mm512_fmadd_pd(EXP_POLY_5, m, EXP_POLY_4);

    __m512d m2 = _mm512_mul_pd(m, m);
    z8 = _mm512_fmadd_pd(z10, m2, z8);
    z4 = _mm512_fmadd_pd(z6, m2, z4); 
    
    __m512d m4 = _mm512_mul_pd(m2, m2);
    z4 = _mm512_fmadd_pd(z8, m4, z4);

    t = _mm512_fmadd_pd(z4, m, EXP_POLY_3);
    t = _mm512_fmadd_pd(t, m, EXP_POLY_2);
    t = _mm512_fmadd_pd(t, m, EXP_POLY_1);
    t = _mm512_fmadd_pd(t, m, EXP_POLY_0);
    
    // fast scale
    __m512i i_scale = _mm512_slli_epi64(i, SCALE_D); 
    __m512d z = (__m512d)_mm512_add_epi32(i_scale, (__m512i)t); 

    __m512d abs_a = (__m512d)_mm512_and_si512((__m512i)a, HI_ABS_MASK);

    __mmask8 exp_slowmask = _mm512_cmpgt_epi64_mask((__m512i)abs_a, (__m512i)UPPERBOUND_1);

//    if (exp_slowmask) {
//        return __pgm_exp_d_vec512_slowpath(a, i, t, z);
//    }


//    int exp_slowmask = _mm512_movemask_pd(_mm512_cmp_pd(abs_a, UPPERBOUND_1, _CMP_GE_OS));
    
    if (__builtin_expect(exp_slowmask, 0)) {
        return __pgm_exp_d_vec512_slowpath(a, i, t, z);
    }

    return z;
} 
