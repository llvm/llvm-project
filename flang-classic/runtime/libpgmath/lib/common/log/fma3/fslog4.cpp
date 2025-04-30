
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
#include "fslog_defs.h"

extern "C" __m128 __fvs_log_fma3(__m128);

__m128 __fvs_log_fma3(__m128 a) {
    __m128 const LOG_C1_VEC = _mm_set1_ps(LOG_C1);
    __m128 const LOG_C2_VEC = _mm_set1_ps(LOG_C2);
    __m128 const LOG_C3_VEC = _mm_set1_ps(LOG_C3);
    __m128 const LOG_C4_VEC = _mm_set1_ps(LOG_C4);
    __m128 const LOG_C5_VEC = _mm_set1_ps(LOG_C5);
    __m128 const LOG_C6_VEC = _mm_set1_ps(LOG_C6);
    __m128 const LOG_C7_VEC = _mm_set1_ps(LOG_C7);
    __m128 const LOG_C8_VEC = _mm_set1_ps(LOG_C8);
    __m128 const LOG_C9_VEC = _mm_set1_ps(LOG_C9);
    __m128 const LOG_CA_VEC = _mm_set1_ps(LOG_CA);

    __m128i const CANONICAL_NAN_VEC = _mm_set1_epi32(CANONICAL_NAN);
    __m128i const MINUS_INF_VEC = _mm_set1_epi32(MINUS_INF);
    __m128i const NAN_INF_MASK_VEC = _mm_set1_epi32(NAN_INF_MASK);

    __m128 const PARTITION_CONST_VEC = _mm_set1_ps(PARTITION_CONST);
    __m128 const TWO_TO_M126_F_VEC = _mm_set1_ps(TWO_TO_M126_F);
    __m128 const TWO_TO_24_F_VEC = _mm_set1_ps(TWO_TO_24_F);

    __m128 const ONE_VEC = _mm_set1_ps(1.0f);
    __m128 const F24_VEC = _mm_set1_ps(U24);
    __m128i const BIT_MASK2_VEC = _mm_set1_epi32(BIT_MASK2);
    __m128i const OFFSET_VEC = _mm_set1_epi32(OFFSET);
    __m128i exp_offset_vec = _mm_set1_epi32(EXP_OFFSET);
    
    __m128 const FLT2INT_CVT = _mm_set1_ps(12582912.0f);
    __m128 FLT2INT_CVT_BIAS = _mm_set1_ps(12582912.0f + 126.0f);
    
    __m128 mask = _mm_cmp_ps(a, TWO_TO_M126_F_VEC, _CMP_LT_OS);
    __m128 fix = _mm_blendv_ps(ONE_VEC, TWO_TO_24_F_VEC, mask);
    a = _mm_mul_ps(a, fix);
    FLT2INT_CVT_BIAS = _mm_add_ps(FLT2INT_CVT_BIAS, _mm_and_ps(mask, F24_VEC));

    __m128 tmpm;
    __m128 spec;

    mask = _mm_cmp_ps(a, _mm_set1_ps(0.0f), _CMP_LT_OS);
    spec = _mm_and_ps((__m128)CANONICAL_NAN_VEC, mask);
        
    mask = _mm_cmp_ps(a, _mm_set1_ps(0.0f), _CMP_EQ_OS);
    tmpm = _mm_and_ps(mask, (__m128)MINUS_INF_VEC);
    spec = _mm_or_ps(tmpm, spec);
    
    mask = _mm_cmp_ps(a, (__m128)NAN_INF_MASK_VEC, _CMP_EQ_OS);
    tmpm = _mm_and_ps(mask, a);
    spec = _mm_or_ps(tmpm,spec);
    mask = _mm_cmp_ps(a, a, _CMP_NEQ_UQ);
    tmpm = _mm_and_ps(mask, _mm_add_ps(a,a));
    spec = _mm_or_ps(tmpm,spec);

    __m128 e = (__m128)_mm_srli_epi32((__m128i)a, 23);
           e = (__m128)_mm_add_epi32((__m128i)e, (__m128i)FLT2INT_CVT);
           e = _mm_sub_ps(e, FLT2INT_CVT_BIAS);

    __m128 m = _mm_and_ps((__m128)BIT_MASK2_VEC, a);
           m = (__m128)_mm_add_epi32((__m128i)m, OFFSET_VEC);
    
    __m128 mask_shift = _mm_cmp_ps(m, PARTITION_CONST_VEC, _CMP_LT_OS);
    
    e = _mm_sub_ps(e, _mm_and_ps(mask_shift, _mm_set1_ps(1.0f)));
    m = _mm_add_ps(m, _mm_and_ps(mask_shift, m));
    m = _mm_sub_ps(m, _mm_set1_ps(1.0f));
    
    __m128 const LN2 = _mm_set1_ps(0x1.62E43p-01);
    e = _mm_mul_ps(e, LN2);

    __m128 t =                       LOG_CA_VEC;
           t = _mm_fmadd_ps(t, m, LOG_C9_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C8_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C7_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C6_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C5_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C4_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C3_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C2_VEC);
           t = _mm_fmadd_ps(t, m, LOG_C1_VEC);

    __m128 m2 = _mm_mul_ps(m, m);
           t = _mm_fmadd_ps(t, m2, m);
           t = _mm_add_ps(t, e);
           t = _mm_add_ps(t, spec); 

    return t;
}

