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
#include "fdlog_defs.h"
#include <stdio.h>

extern "C" __m512d FCN_AVX512(__fvd_log_fma3)(__m512d);


// casts int to double
inline
__m512d __internal_fast_int2dbl(__m512i a)
{
    __m512i const INT2DBL_HI = _mm512_set1_epi64(INT2DBL_HI_D);
    __m512i const INT2DBL_LO = _mm512_set1_epi64(INT2DBL_LO_D);
    __m512d const INT2DBL    = (__m512d)_mm512_set1_epi64(INT2DBL_D);

    __m512i t = _mm512_xor_si512(INT2DBL_LO, a);
    t = _mm512_mask_blend_epi32(0x5555, INT2DBL_HI, t);
    return _mm512_sub_pd((__m512d)t, INT2DBL);
}

// special cases for log
static __m512d __attribute__ ((noinline)) __pgm_log_d_vec512_special_cases(__m512d const a, __m512d z)
{
    __m512d const ZERO         = _mm512_set1_pd(ZERO_D);
    __m512i const ALL_ONES_EXPONENT = _mm512_set1_epi64(ALL_ONES_EXPONENT_D);
    __m512d const NAN_VAL   = (__m512d)_mm512_set1_epi64(NAN_VAL_D);
    __m512d const NEG_INF  = (__m512d)_mm512_set1_epi64(NEG_INF_D);


    __m512i detect_inf_nan = (__m512i)_mm512_sub_pd(a, a); 
    __m512d inf_nan_mask = (__m512d)_MM512_CMPEQ_EPI64(_mm512_and_si512(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);
   
    // inf + inf = inf = log(inf). nan + nan = nan = log(nan).
    __m512i inf_nan = (__m512i)_mm512_add_pd(a, a);
    z = _MM512_BLENDV_PD(z, (__m512d)inf_nan, inf_nan_mask); 
    
    __m512d non_positive_mask = _MM512_CMP_PD(a, ZERO, _CMP_LT_OQ);
    // log(negative number) = NaN
    z = _MM512_BLENDV_PD(z, NAN_VAL, non_positive_mask);

    // log(0) = -inf
    __m512d zero_mask = _MM512_CMP_PD(a, ZERO, _CMP_EQ_OQ);
    z = _MM512_BLENDV_PD(z, NEG_INF, zero_mask);
     
    return z;
}

__m512d FCN_AVX512(__fvd_log_fma3)(__m512d const a)
{

     __m512d const HI_CONST_1   = (__m512d)_mm512_set1_epi64(HI_CONST_1_D);
     __m512d const HI_CONST_2   = (__m512d)_mm512_set1_epi64(HI_CONST_2_D);
     __m512i const HALFIFIER    = _mm512_set1_epi64(HALFIFIER_D);
     __m512i const HI_THRESH    = _mm512_set1_epi64(HI_THRESH_D);
     __m512d const ONE_F        = _mm512_set1_pd(ONE_F_D);
     __m512d const ZERO         = _mm512_set1_pd(ZERO_D);

     __m512d const LN2_HI       = _mm512_set1_pd(LN2_HI_D);
     __m512d const LN2_LO       = _mm512_set1_pd(LN2_LO_D);

     __m512i const HI_MASK      = _mm512_set1_epi64(HI_MASK_D);
     __m512i const ONE          = _mm512_set1_epi64(ONE_D);

     __m512i const TEN_23      = _mm512_set1_epi64(TEN_23_D);
     __m512i const ALL_ONES_EXPONENT = _mm512_set1_epi64(ALL_ONES_EXPONENT_D);

    __m512d const LOG_C1_VEC = _mm512_set1_pd(   LOG_C1_VEC_D    );
    __m512d const LOG_C2_VEC = _mm512_set1_pd(   LOG_C2_VEC_D    );
    __m512d const LOG_C3_VEC = _mm512_set1_pd(   LOG_C3_VEC_D    );
    __m512d const LOG_C4_VEC = _mm512_set1_pd(   LOG_C4_VEC_D    );
    __m512d const LOG_C5_VEC = _mm512_set1_pd(   LOG_C5_VEC_D    );
    __m512d const LOG_C6_VEC = _mm512_set1_pd(   LOG_C6_VEC_D    );
    __m512d const LOG_C7_VEC = _mm512_set1_pd(   LOG_C7_VEC_D    );
    __m512d const LOG_C8_VEC = _mm512_set1_pd(   LOG_C8_VEC_D    );
    __m512d const LOG_C9_VEC = _mm512_set1_pd(   LOG_C9_VEC_D    );
    __m512d const LOG_C10_VEC = _mm512_set1_pd(  LOG_C10_VEC_D   );
    __m512d const LOG_C11_VEC = _mm512_set1_pd(  LOG_C11_VEC_D   );
    __m512d const LOG_C12_VEC = _mm512_set1_pd(  LOG_C12_VEC_D   );
    __m512d const LOG_C13_VEC = _mm512_set1_pd(  LOG_C13_VEC_D   );
    __m512d const LOG_C14_VEC = _mm512_set1_pd(  LOG_C14_VEC_D   );
    __m512d const LOG_C15_VEC = _mm512_set1_pd(  LOG_C15_VEC_D   );
    __m512d const LOG_C16_VEC = _mm512_set1_pd(  LOG_C16_VEC_D   );
    __m512d const LOG_C17_VEC = _mm512_set1_pd(  LOG_C17_VEC_D   );
    __m512d const LOG_C18_VEC = _mm512_set1_pd(  LOG_C18_VEC_D   );
    __m512d const LOG_C19_VEC = _mm512_set1_pd(  LOG_C19_VEC_D   );
    __m512d const LOG_C20_VEC = _mm512_set1_pd(  LOG_C20_VEC_D   );
    __m512d const LOG_C21_VEC = _mm512_set1_pd(  LOG_C21_VEC_D   );
    __m512d const LOG_C22_VEC = _mm512_set1_pd(  LOG_C22_VEC_D   );
    __m512d const LOG_C23_VEC = _mm512_set1_pd(  LOG_C23_VEC_D   );
    __m512d const LOG_C24_VEC = _mm512_set1_pd(  LOG_C24_VEC_D   );

    __m512d a_mut, m, f;
    __m512i expo, expo_plus1;
    __m512d thresh_mask;

    // isolate mantissa
    a_mut = _MM512_AND_PD(a, HI_CONST_1);
    a_mut = _MM512_OR_PD(a_mut, HI_CONST_2);

    // magic trick to improve accuracy (divide mantissa by 2 and increase exponent by 1)
    thresh_mask = _MM512_CMP_PD(a_mut, (__m512d)HI_THRESH, _CMP_GT_OS);
    m = (__m512d)_mm512_sub_epi32((__m512i)a_mut, HALFIFIER); 
    m = _MM512_BLENDV_PD(a_mut, m, thresh_mask);   

    // compute exponent
    expo = _mm512_srli_epi64((__m512i)a, D52_D);
    expo = _mm512_sub_epi64(expo, TEN_23);
    expo_plus1 = _mm512_add_epi64(expo, ONE);     
    expo = (__m512i)_MM512_BLENDV_PD((__m512d)expo, (__m512d)expo_plus1, thresh_mask);

    // computing polynomial for log(1+m)
    m = _mm512_sub_pd(m, ONE_F);

    // estrin scheme for highest 16 terms, then estrin again for the next 8. Finally finish off with horner.
    __m512d z9  = _mm512_fmadd_pd(LOG_C10_VEC, m, LOG_C9_VEC);
    __m512d z11 = _mm512_fmadd_pd(LOG_C12_VEC, m, LOG_C11_VEC);
    __m512d z13 = _mm512_fmadd_pd(LOG_C14_VEC, m, LOG_C13_VEC);
    __m512d z15 = _mm512_fmadd_pd(LOG_C16_VEC, m, LOG_C15_VEC);
    __m512d z17 = _mm512_fmadd_pd(LOG_C18_VEC, m, LOG_C17_VEC);
    __m512d z19 = _mm512_fmadd_pd(LOG_C20_VEC, m, LOG_C19_VEC);
    __m512d z21 = _mm512_fmadd_pd(LOG_C22_VEC, m, LOG_C21_VEC);
    __m512d z23 = _mm512_fmadd_pd(LOG_C24_VEC, m, LOG_C23_VEC);

    __m512d m2 = _mm512_mul_pd(m, m);
    z9  = _mm512_fmadd_pd(z11, m2, z9);
    z13 = _mm512_fmadd_pd(z15, m2, z13);
    z17 = _mm512_fmadd_pd(z19, m2, z17);
    z21 = _mm512_fmadd_pd(z23, m2, z21);

    __m512d m4 = _mm512_mul_pd(m2, m2);
    z9  = _mm512_fmadd_pd(z13, m4, z9);
    z17 = _mm512_fmadd_pd(z21, m4, z17);

    __m512d m8 = _mm512_mul_pd(m4, m4);
    z9 = _mm512_fmadd_pd(z17, m8, z9);
  
    // estrin for the next 8 terms
    __m512d z8 = _mm512_fmadd_pd(z9, m, LOG_C8_VEC);
    __m512d z6 = _mm512_fmadd_pd(LOG_C7_VEC, m, LOG_C6_VEC);
    __m512d z4 = _mm512_fmadd_pd(LOG_C5_VEC, m, LOG_C4_VEC);
    __m512d z2 = _mm512_fmadd_pd(LOG_C3_VEC, m, LOG_C2_VEC);
    
    z6 = _mm512_fmadd_pd(z8, m2, z6);
    z2 = _mm512_fmadd_pd(z4, m2, z2);
    __m512d z = _mm512_fmadd_pd(z6, m4, z2);

    // finish computation with horner
    z = _mm512_fmadd_pd(z, m, LOG_C1_VEC);
    z = _mm512_mul_pd(z, m);

    f = __internal_fast_int2dbl(expo);
    z = _mm512_fmadd_pd(f, LN2_HI, z);
     
    // compute special cases (inf, NaN, negative, 0)
    __m512i detect_inf_nan = (__m512i)_mm512_sub_pd(a, a);
    __m512i detect_non_positive = (__m512i)_MM512_CMP_PD(a, ZERO, _CMP_LE_OQ);
    __m512i inf_nan_mask = _MM512_CMPEQ_EPI64(_mm512_and_si512(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);

    int specMask = _MM512_MOVEMASK_PD((__m512d)_mm512_or_si512(detect_non_positive, inf_nan_mask));
    if(__builtin_expect(specMask, 0)) {
        return __pgm_log_d_vec512_special_cases(a, z);
    }
    return z;
} 
