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
#include "fdlog_defs.h"

extern "C" __m256d __fvd_log_fma3_256(__m256d);


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

// special cases for log
__m256d __attribute__ ((noinline)) __pgm_log_d_vec256_special_cases(__m256d const a, __m256d z)
{
    __m256d const ZERO         = _mm256_set1_pd(ZERO_D);
    __m256i const ALL_ONES_EXPONENT = _mm256_set1_epi64x(ALL_ONES_EXPONENT_D);
    __m256d const NAN_VAL   = (__m256d)_mm256_set1_epi64x(NAN_VAL_D);
    __m256d const NEG_INF  = (__m256d)_mm256_set1_epi64x(NEG_INF_D);


    __m256i detect_inf_nan = (__m256i)_mm256_sub_pd(a, a); 
    __m256d inf_nan_mask = (__m256d)_mm256_cmpeq_epi64(_mm256_and_si256(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);
   
    // inf + inf = inf = log(inf). nan + nan = nan = log(nan).
    __m256i inf_nan = (__m256i)_mm256_add_pd(a, a);
    z = _mm256_blendv_pd(z, (__m256d)inf_nan, inf_nan_mask); 
    
    __m256d non_positive_mask = _mm256_cmp_pd(a, ZERO, _CMP_LT_OQ);
    // log(negative number) = NaN
    z = _mm256_blendv_pd(z, NAN_VAL, non_positive_mask);

    // log(0) = -inf
    __m256d zero_mask = _mm256_cmp_pd(a, ZERO, _CMP_EQ_OQ);
    z = _mm256_blendv_pd(z, NEG_INF, zero_mask);
     
    return z;
}

__m256d __fvd_log_fma3_256(__m256d const a)
{

     __m256d const HI_CONST_1   = (__m256d)_mm256_set1_epi64x(HI_CONST_1_D);
     __m256d const HI_CONST_2   = (__m256d)_mm256_set1_epi64x(HI_CONST_2_D);
     __m256i const HALFIFIER    = _mm256_set1_epi64x(HALFIFIER_D);
     __m256i const HI_THRESH    = _mm256_set1_epi64x(HI_THRESH_D);
     __m256d const ONE_F        = _mm256_set1_pd(ONE_F_D);
     __m256d const ZERO         = _mm256_set1_pd(ZERO_D);

     __m256d const LN2_HI       = _mm256_set1_pd(LN2_HI_D);
     __m256d const LN2_LO       = _mm256_set1_pd(LN2_LO_D);

     __m256i const HI_MASK      = _mm256_set1_epi64x(HI_MASK_D);
     __m256i const ONE          = _mm256_set1_epi64x(ONE_D);

     __m256i const TEN_23      = _mm256_set1_epi64x(TEN_23_D);
     __m256i const ALL_ONES_EXPONENT = _mm256_set1_epi64x(ALL_ONES_EXPONENT_D);

    __m256d const LOG_C1_VEC = _mm256_set1_pd(   LOG_C1_VEC_D    );
    __m256d const LOG_C2_VEC = _mm256_set1_pd(   LOG_C2_VEC_D    );
    __m256d const LOG_C3_VEC = _mm256_set1_pd(   LOG_C3_VEC_D    );
    __m256d const LOG_C4_VEC = _mm256_set1_pd(   LOG_C4_VEC_D    );
    __m256d const LOG_C5_VEC = _mm256_set1_pd(   LOG_C5_VEC_D    );
    __m256d const LOG_C6_VEC = _mm256_set1_pd(   LOG_C6_VEC_D    );
    __m256d const LOG_C7_VEC = _mm256_set1_pd(   LOG_C7_VEC_D    );
    __m256d const LOG_C8_VEC = _mm256_set1_pd(   LOG_C8_VEC_D    );
    __m256d const LOG_C9_VEC = _mm256_set1_pd(   LOG_C9_VEC_D    );
    __m256d const LOG_C10_VEC = _mm256_set1_pd(  LOG_C10_VEC_D   );
    __m256d const LOG_C11_VEC = _mm256_set1_pd(  LOG_C11_VEC_D   );
    __m256d const LOG_C12_VEC = _mm256_set1_pd(  LOG_C12_VEC_D   );
    __m256d const LOG_C13_VEC = _mm256_set1_pd(  LOG_C13_VEC_D   );
    __m256d const LOG_C14_VEC = _mm256_set1_pd(  LOG_C14_VEC_D   );
    __m256d const LOG_C15_VEC = _mm256_set1_pd(  LOG_C15_VEC_D   );
    __m256d const LOG_C16_VEC = _mm256_set1_pd(  LOG_C16_VEC_D   );
    __m256d const LOG_C17_VEC = _mm256_set1_pd(  LOG_C17_VEC_D   );
    __m256d const LOG_C18_VEC = _mm256_set1_pd(  LOG_C18_VEC_D   );
    __m256d const LOG_C19_VEC = _mm256_set1_pd(  LOG_C19_VEC_D   );
    __m256d const LOG_C20_VEC = _mm256_set1_pd(  LOG_C20_VEC_D   );
    __m256d const LOG_C21_VEC = _mm256_set1_pd(  LOG_C21_VEC_D   );
    __m256d const LOG_C22_VEC = _mm256_set1_pd(  LOG_C22_VEC_D   );
    __m256d const LOG_C23_VEC = _mm256_set1_pd(  LOG_C23_VEC_D   );
    __m256d const LOG_C24_VEC = _mm256_set1_pd(  LOG_C24_VEC_D   );

    __m256d a_mut, m, f;
    __m256i expo, expo_plus1;
    __m256d thresh_mask;

    // isolate mantissa
    a_mut = _mm256_and_pd(a, HI_CONST_1);
    a_mut = _mm256_or_pd(a_mut, HI_CONST_2);

    // magic trick to improve accuracy (divide mantissa by 2 and increase exponent by 1)
    thresh_mask = _mm256_cmp_pd(a_mut, (__m256d)HI_THRESH, _CMP_GT_OS);
    m = (__m256d)_mm256_sub_epi32((__m256i)a_mut, HALFIFIER); 
    m = _mm256_blendv_pd(a_mut, m, thresh_mask);   

    // compute exponent
    expo = _mm256_srli_epi64((__m256i)a, D52_D);
    expo = _mm256_sub_epi64(expo, TEN_23);
    expo_plus1 = _mm256_add_epi64(expo, ONE);     
    expo = (__m256i)_mm256_blendv_pd((__m256d)expo, (__m256d)expo_plus1, thresh_mask);

    // computing polynomial for log(1+m)
    m = _mm256_sub_pd(m, ONE_F);

    // estrin scheme for highest 16 terms, then estrin again for the next 8. Finally finish off with horner.
    __m256d z9  = _mm256_fmadd_pd(LOG_C10_VEC, m, LOG_C9_VEC);
    __m256d z11 = _mm256_fmadd_pd(LOG_C12_VEC, m, LOG_C11_VEC);
    __m256d z13 = _mm256_fmadd_pd(LOG_C14_VEC, m, LOG_C13_VEC);
    __m256d z15 = _mm256_fmadd_pd(LOG_C16_VEC, m, LOG_C15_VEC);
    __m256d z17 = _mm256_fmadd_pd(LOG_C18_VEC, m, LOG_C17_VEC);
    __m256d z19 = _mm256_fmadd_pd(LOG_C20_VEC, m, LOG_C19_VEC);
    __m256d z21 = _mm256_fmadd_pd(LOG_C22_VEC, m, LOG_C21_VEC);
    __m256d z23 = _mm256_fmadd_pd(LOG_C24_VEC, m, LOG_C23_VEC);

    __m256d m2 = _mm256_mul_pd(m, m);
    z9  = _mm256_fmadd_pd(z11, m2, z9);
    z13 = _mm256_fmadd_pd(z15, m2, z13);
    z17 = _mm256_fmadd_pd(z19, m2, z17);
    z21 = _mm256_fmadd_pd(z23, m2, z21);

    __m256d m4 = _mm256_mul_pd(m2, m2);
    z9  = _mm256_fmadd_pd(z13, m4, z9);
    z17 = _mm256_fmadd_pd(z21, m4, z17);

    __m256d m8 = _mm256_mul_pd(m4, m4);
    z9 = _mm256_fmadd_pd(z17, m8, z9);
  
    // estrin for the next 8 terms
    __m256d z8 = _mm256_fmadd_pd(z9, m, LOG_C8_VEC);
    __m256d z6 = _mm256_fmadd_pd(LOG_C7_VEC, m, LOG_C6_VEC);
    __m256d z4 = _mm256_fmadd_pd(LOG_C5_VEC, m, LOG_C4_VEC);
    __m256d z2 = _mm256_fmadd_pd(LOG_C3_VEC, m, LOG_C2_VEC);
    
    z6 = _mm256_fmadd_pd(z8, m2, z6);
    z2 = _mm256_fmadd_pd(z4, m2, z2);
    __m256d z = _mm256_fmadd_pd(z6, m4, z2);

    // finish computation with horner
    z = _mm256_fmadd_pd(z, m, LOG_C1_VEC);
    z = _mm256_mul_pd(z, m);

    f = __internal_fast_int2dbl(expo);
    z = _mm256_fmadd_pd(f, LN2_HI, z);
     
    // compute special cases (inf, NaN, negative, 0)
    __m256i detect_inf_nan = (__m256i)_mm256_sub_pd(a, a);
    __m256i detect_non_positive = (__m256i)_mm256_cmp_pd(a, ZERO, _CMP_LE_OQ);
    __m256i inf_nan_mask = _mm256_cmpeq_epi64(_mm256_and_si256(detect_inf_nan, ALL_ONES_EXPONENT), ALL_ONES_EXPONENT);

    int specMask = _mm256_movemask_pd((__m256d)_mm256_or_si256(detect_non_positive, inf_nan_mask));
    if(__builtin_expect(specMask, 0)) {
        return __pgm_log_d_vec256_special_cases(a, z);
    }
    return z;
} 
