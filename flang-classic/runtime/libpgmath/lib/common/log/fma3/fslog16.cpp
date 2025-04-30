
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
#include "fslog_defs.h"

extern "C" __m512 FCN_AVX512(__fvs_log_fma3)(__m512);

__m512 FCN_AVX512(__fvs_log_fma3)(__m512 a) {
    __m512 const LOG_C1_VEC = _mm512_set1_ps(LOG_C1);
    __m512 const LOG_C2_VEC = _mm512_set1_ps(LOG_C2);
    __m512 const LOG_C3_VEC = _mm512_set1_ps(LOG_C3);
    __m512 const LOG_C4_VEC = _mm512_set1_ps(LOG_C4);
    __m512 const LOG_C5_VEC = _mm512_set1_ps(LOG_C5);
    __m512 const LOG_C6_VEC = _mm512_set1_ps(LOG_C6);
    __m512 const LOG_C7_VEC = _mm512_set1_ps(LOG_C7);
    __m512 const LOG_C8_VEC = _mm512_set1_ps(LOG_C8);
    __m512 const LOG_C9_VEC = _mm512_set1_ps(LOG_C9);
    __m512 const LOG_CA_VEC = _mm512_set1_ps(LOG_CA);

    __m512i const CANONICAL_NAN_VEC = _mm512_set1_epi32(CANONICAL_NAN);
    __m512i const MINUS_INF_VEC = _mm512_set1_epi32(MINUS_INF);
    __m512i const NAN_INF_MASK_VEC = _mm512_set1_epi32(NAN_INF_MASK);

    __m512 const PARTITION_CONST_VEC = _mm512_set1_ps(PARTITION_CONST);
    __m512 const TWO_TO_M126_F_VEC = _mm512_set1_ps(TWO_TO_M126_F);
    __m512 const TWO_TO_24_F_VEC = _mm512_set1_ps(TWO_TO_24_F);
    __m512 const ONE_VEC = _mm512_set1_ps(1.0f);
    __m512 const F24_VEC = _mm512_set1_ps(U24);
    __m512i const BIT_MASK2_VEC = _mm512_set1_epi32(BIT_MASK2);
    __m512i const OFFSET_VEC = _mm512_set1_epi32(OFFSET);
    __m512i exp_offset_vec = _mm512_set1_epi32(EXP_OFFSET);
    
    __m512 const FLT2INT_CVT = _mm512_set1_ps(12582912.0f);
    __m512 FLT2INT_CVT_BIAS = _mm512_set1_ps(12582912.0f + 126.0f);
    
    __m512 mask = _MM512_CMP_PS(a, TWO_TO_M126_F_VEC, _CMP_LT_OS);
    __m512 fix = _MM512_BLENDV_PS(ONE_VEC, TWO_TO_24_F_VEC, mask);
    a = _mm512_mul_ps(a, fix);
    FLT2INT_CVT_BIAS = _mm512_add_ps(FLT2INT_CVT_BIAS, _MM512_AND_PS(mask, F24_VEC));

    __m512 tmpm;
    __m512 spec;

    mask = _MM512_CMP_PS(a, _mm512_set1_ps(0.0f), _CMP_LT_OS);
    spec = _MM512_AND_PS((__m512)CANONICAL_NAN_VEC, mask);
        
    mask = _MM512_CMP_PS(a, _mm512_set1_ps(0.0f), _CMP_EQ_OS);
    tmpm = _MM512_AND_PS(mask, (__m512)MINUS_INF_VEC);
    spec = _MM512_OR_PS(tmpm, spec);
    
    mask = _MM512_CMP_PS(a, (__m512)NAN_INF_MASK_VEC, _CMP_EQ_OS);
    tmpm = _MM512_AND_PS(mask, a);
    spec = _MM512_OR_PS(tmpm,spec);
    mask = _MM512_CMP_PS(a, a, 4);
    tmpm = _MM512_AND_PS(mask, _mm512_add_ps(a,a));
    spec = _MM512_OR_PS(tmpm,spec);

    __m512 e = (__m512)_mm512_srli_epi32((__m512i)a, 23);
           e = (__m512)_mm512_add_epi32((__m512i)e, (__m512i)FLT2INT_CVT);
           e = _mm512_sub_ps(e, FLT2INT_CVT_BIAS);

    __m512 m = _MM512_AND_PS((__m512)BIT_MASK2_VEC, a);
           m = (__m512)_mm512_add_epi32((__m512i)m, OFFSET_VEC);
    
    __m512 mask_shift = _MM512_CMP_PS(m, PARTITION_CONST_VEC, _CMP_LT_OS);
    
    e = _mm512_sub_ps(e, _MM512_AND_PS(mask_shift, _mm512_set1_ps(1.0f)));
    m = _mm512_add_ps(m, _MM512_AND_PS(mask_shift, m));
    m = _mm512_sub_ps(m, _mm512_set1_ps(1.0f));
    
    __m512 const LN2 = _mm512_set1_ps(0x1.62E43p-01);
    e = _mm512_mul_ps(e, LN2);

    __m512 t =                       LOG_CA_VEC;
           t = _mm512_fmadd_ps(t, m, LOG_C9_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C8_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C7_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C6_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C5_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C4_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C3_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C2_VEC);
           t = _mm512_fmadd_ps(t, m, LOG_C1_VEC);

    __m512 m2 = _mm512_mul_ps(m, m);
           t = _mm512_fmadd_ps(t, m2, m);
           t = _mm512_add_ps(t, e);
           t = _mm512_add_ps(t, spec); 

    return t;
}


