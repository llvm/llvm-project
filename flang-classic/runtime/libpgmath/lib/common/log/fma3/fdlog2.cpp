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
#include "fdlog_defs.h"

extern "C" __m128d __fvd_log_fma3(__m128d);


// casts int to double
inline
__m128d __internal_fast_int2dbl(__m128i a)
{
    __m128i const INT2DBL_HI = _mm_set1_epi64x(INT2DBL_HI_D);
    __m128i const INT2DBL_LO = _mm_set1_epi64x(INT2DBL_LO_D);
    __m128d const INT2DBL    = (__m128d)_mm_set1_epi64x(INT2DBL_D);

    __m128i t = _mm_xor_si128(INT2DBL_LO, a);
    t = _mm_blend_epi32(INT2DBL_HI, t, 0x5); 
    return _mm_sub_pd((__m128d)t, INT2DBL);
}

// special cases for log
__m128d __attribute__ ((noinline)) __pgm_log_d_vec128_special_cases(__m128d const a, __m128d z)
{
    __m128d const ZERO         = _mm_set1_pd(ZERO_D);
    __m128i const ALL_ONES_EXPONENT = _mm_set1_epi64x(ALL_ONES_EXPONENT_D);
    __m128d const NAN_VAL   = (__m128d)_mm_set1_epi64x(NAN_VAL_D);
    __m128d const NEG_INF  = (__m128d)_mm_set1_epi64x(NEG_INF_D);


    __m128i detect_inf_nan = (__m128i)_mm_sub_pd(a, a); 
    __m128d inf_nan_mask = (__m128d)_mm_cmpeq_epi64(_mm_and_si128(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);
   
    // inf + inf = inf = log(inf). nan + nan = nan = log(nan).
    __m128i inf_nan = (__m128i)_mm_add_pd(a, a);
    z = _mm_blendv_pd(z, (__m128d)inf_nan, inf_nan_mask); 
    
    __m128d non_positive_mask = _mm_cmp_pd(a, ZERO, _CMP_LT_OQ);
    // log(negative number) = NaN
    z = _mm_blendv_pd(z, NAN_VAL, non_positive_mask);

    // log(0) = -inf
    __m128d zero_mask = _mm_cmp_pd(a, ZERO, _CMP_EQ_OQ);
    z = _mm_blendv_pd(z, NEG_INF, zero_mask);
     
    return z;
}
__m128d __fvd_log_fma3(__m128d const a)
{

     __m128d const HI_CONST_1   = (__m128d)_mm_set1_epi64x(HI_CONST_1_D);
     __m128d const HI_CONST_2   = (__m128d)_mm_set1_epi64x(HI_CONST_2_D);
     __m128i const HALFIFIER    = _mm_set1_epi64x(HALFIFIER_D);
     __m128i const HI_THRESH    = _mm_set1_epi64x(HI_THRESH_D);
     __m128d const ONE_F        = _mm_set1_pd(ONE_F_D);
     __m128d const ZERO         = _mm_set1_pd(ZERO_D);

     __m128d const LN2_HI       = _mm_set1_pd(LN2_HI_D);
     __m128d const LN2_LO       = _mm_set1_pd(LN2_LO_D);

     __m128i const HI_MASK      = _mm_set1_epi64x(HI_MASK_D);
     __m128i const ONE          = _mm_set1_epi64x(ONE_D);

     __m128i const TEN_23      = _mm_set1_epi64x(TEN_23_D);
     __m128i const ALL_ONES_EXPONENT = _mm_set1_epi64x(ALL_ONES_EXPONENT_D);

    __m128d const LOG_C1_VEC = _mm_set1_pd(   LOG_C1_VEC_D    );
    __m128d const LOG_C2_VEC = _mm_set1_pd(   LOG_C2_VEC_D    );
    __m128d const LOG_C3_VEC = _mm_set1_pd(   LOG_C3_VEC_D    );
    __m128d const LOG_C4_VEC = _mm_set1_pd(   LOG_C4_VEC_D    );
    __m128d const LOG_C5_VEC = _mm_set1_pd(   LOG_C5_VEC_D    );
    __m128d const LOG_C6_VEC = _mm_set1_pd(   LOG_C6_VEC_D    );
    __m128d const LOG_C7_VEC = _mm_set1_pd(   LOG_C7_VEC_D    );
    __m128d const LOG_C8_VEC = _mm_set1_pd(   LOG_C8_VEC_D    );
    __m128d const LOG_C9_VEC = _mm_set1_pd(   LOG_C9_VEC_D    );
    __m128d const LOG_C10_VEC = _mm_set1_pd(  LOG_C10_VEC_D   );
    __m128d const LOG_C11_VEC = _mm_set1_pd(  LOG_C11_VEC_D   );
    __m128d const LOG_C12_VEC = _mm_set1_pd(  LOG_C12_VEC_D   );
    __m128d const LOG_C13_VEC = _mm_set1_pd(  LOG_C13_VEC_D   );
    __m128d const LOG_C14_VEC = _mm_set1_pd(  LOG_C14_VEC_D   );
    __m128d const LOG_C15_VEC = _mm_set1_pd(  LOG_C15_VEC_D   );
    __m128d const LOG_C16_VEC = _mm_set1_pd(  LOG_C16_VEC_D   );
    __m128d const LOG_C17_VEC = _mm_set1_pd(  LOG_C17_VEC_D   );
    __m128d const LOG_C18_VEC = _mm_set1_pd(  LOG_C18_VEC_D   );
    __m128d const LOG_C19_VEC = _mm_set1_pd(  LOG_C19_VEC_D   );
    __m128d const LOG_C20_VEC = _mm_set1_pd(  LOG_C20_VEC_D   );
    __m128d const LOG_C21_VEC = _mm_set1_pd(  LOG_C21_VEC_D   );
    __m128d const LOG_C22_VEC = _mm_set1_pd(  LOG_C22_VEC_D   );
    __m128d const LOG_C23_VEC = _mm_set1_pd(  LOG_C23_VEC_D   );
    __m128d const LOG_C24_VEC = _mm_set1_pd(  LOG_C24_VEC_D   );


    __m128d a_mut, m, f;
    __m128i expo, expo_plus1;
    __m128d thresh_mask;

    // isolate mantissa
    a_mut = _mm_and_pd(a, HI_CONST_1);
    a_mut = _mm_or_pd(a_mut, HI_CONST_2);

    // magic trick to improve accuracy (divide mantissa by 2 and increase exponent by 1)
    thresh_mask = _mm_cmp_pd(a_mut, (__m128d)HI_THRESH, _CMP_GT_OS);
    m = (__m128d)_mm_sub_epi32((__m128i)a_mut, HALFIFIER); 
    m = _mm_blendv_pd(a_mut, m, thresh_mask);   

    // compute exponent
    expo = _mm_srli_epi64((__m128i)a, D52_D);
    expo = _mm_sub_epi64(expo, TEN_23);
    expo_plus1 = _mm_add_epi64(expo, ONE);     
    expo = (__m128i)_mm_blendv_pd((__m128d)expo, (__m128d)expo_plus1, thresh_mask);

    // computing polynomial for log(1+m)
    m = _mm_sub_pd(m, ONE_F);

    // estrin scheme for highest 16 terms, then estrin again for the next 8. Finally finish off with horner.
    __m128d z9  = _mm_fmadd_pd(LOG_C10_VEC, m, LOG_C9_VEC);
    __m128d z11 = _mm_fmadd_pd(LOG_C12_VEC, m, LOG_C11_VEC);
    __m128d z13 = _mm_fmadd_pd(LOG_C14_VEC, m, LOG_C13_VEC);
    __m128d z15 = _mm_fmadd_pd(LOG_C16_VEC, m, LOG_C15_VEC);
    __m128d z17 = _mm_fmadd_pd(LOG_C18_VEC, m, LOG_C17_VEC);
    __m128d z19 = _mm_fmadd_pd(LOG_C20_VEC, m, LOG_C19_VEC);
    __m128d z21 = _mm_fmadd_pd(LOG_C22_VEC, m, LOG_C21_VEC);
    __m128d z23 = _mm_fmadd_pd(LOG_C24_VEC, m, LOG_C23_VEC);

    __m128d m2 = _mm_mul_pd(m, m);
    z9  = _mm_fmadd_pd(z11, m2, z9);
    z13 = _mm_fmadd_pd(z15, m2, z13);
    z17 = _mm_fmadd_pd(z19, m2, z17);
    z21 = _mm_fmadd_pd(z23, m2, z21);

    __m128d m4 = _mm_mul_pd(m2, m2);
    z9  = _mm_fmadd_pd(z13, m4, z9);
    z17 = _mm_fmadd_pd(z21, m4, z17);

    __m128d m8 = _mm_mul_pd(m4, m4);
    z9 = _mm_fmadd_pd(z17, m8, z9);
  
    // estrin for the next 8 terms
    __m128d z8 = _mm_fmadd_pd(z9, m, LOG_C8_VEC);
    __m128d z6 = _mm_fmadd_pd(LOG_C7_VEC, m, LOG_C6_VEC);
    __m128d z4 = _mm_fmadd_pd(LOG_C5_VEC, m, LOG_C4_VEC);
    __m128d z2 = _mm_fmadd_pd(LOG_C3_VEC, m, LOG_C2_VEC);
    
    z6 = _mm_fmadd_pd(z8, m2, z6);
    z2 = _mm_fmadd_pd(z4, m2, z2);
    __m128d z = _mm_fmadd_pd(z6, m4, z2);

    // finish computation with horner
    z = _mm_fmadd_pd(z, m, LOG_C1_VEC);
    z = _mm_mul_pd(z, m);

    f = __internal_fast_int2dbl(expo);
    z = _mm_fmadd_pd(f, LN2_HI, z);
     
    // compute special cases (inf, NaN, negative, 0)
    __m128i detect_inf_nan = (__m128i)_mm_sub_pd(a, a);
    __m128i detect_non_positive = (__m128i)_mm_cmp_pd(a, ZERO, _CMP_LE_OQ);
    __m128i inf_nan_mask = _mm_cmpeq_epi64(_mm_and_si128(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);

#if defined(TARGET_LINUX_POWER)
    int specMask = _vec_any_nz((__m128i)_mm_or_si128(detect_non_positive, inf_nan_mask));
#else
    int specMask = _mm_movemask_pd((__m128d)_mm_or_si128(detect_non_positive, inf_nan_mask));
#endif
    if(__builtin_expect(specMask, 0)) {
        return __pgm_log_d_vec128_special_cases(a, z);
    }
    return z;
} 
