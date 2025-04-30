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
#include "exp_defs.h"

extern "C" __m512 FCN_AVX512(__fvs_exp_fma3)(__m512);
static __m512 __pgm_exp_vec512_slowpath(__m512, __m512i, __m512);

__m512 FCN_AVX512(__fvs_exp_fma3)(__m512 a)
{
    __m512 const EXP_HI_VEC = _mm512_set1_ps(EXP_HI);
    __m512 const EXP_LO_VEC = _mm512_set1_ps(EXP_LO);
    __m512 const EXP_PDN_VEC = _mm512_set1_ps(EXP_PDN);
    __m512 const FLT2INT_CVT_VEC = _mm512_set1_ps(FLT2INT_CVT);
    __m512 const L2E_VEC = _mm512_set1_ps(L2E);
   
    __m512 const SGN_VEC = (__m512)_mm512_set1_epi32(MASK);

    __m512 abs = _MM512_AND_PS(a, SGN_VEC);
    __m512i sp_mask = _MM512_CMPGT_EPI32(_mm512_castps_si512(abs), _mm512_castps_si512(EXP_PDN_VEC)); // zero dla dobrych
    int sp = _MM512_MOVEMASK_EPI32(sp_mask);
    __m512 t = _mm512_fmadd_ps(a, L2E_VEC, FLT2INT_CVT_VEC);
    __m512 tt = _mm512_sub_ps(t, FLT2INT_CVT_VEC);
    __m512 z = _mm512_fnmadd_ps(tt, _mm512_set1_ps(LN2_0), a);
           z = _mm512_fnmadd_ps(tt, _mm512_set1_ps(LN2_1), z);
         
    __m512i exp = _mm512_castps_si512(t);
            exp = _mm512_slli_epi32(exp, 23);

    __m512 zz =                 _mm512_set1_ps(EXP_C7);
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C6));
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C5));
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C4));
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C3));
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C2));
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C1));
    zz = _mm512_fmadd_ps(zz, z, _mm512_set1_ps(EXP_C0));
    __m512 res = (__m512)_mm512_add_epi32(exp, (__m512i)zz);
 
    if (sp)
    {
        res = __pgm_exp_vec512_slowpath(a, exp, zz);       
    }

    return res;
}


static __m512 __pgm_exp_vec512_slowpath(__m512 a, __m512i exp, __m512 zz) {
    __m512 const EXP_HI_VEC = _mm512_set1_ps(EXP_HI);
    __m512 const EXP_LO_VEC = _mm512_set1_ps(EXP_LO);
    __m512i const DNRM_THR_VEC = _mm512_set1_epi32(DNRM_THR);
    __m512i const EXP_BIAS_VEC = _mm512_set1_epi32(EXP_BIAS);
    __m512i const DNRM_SHFT_VEC = _mm512_set1_epi32(DNRM_SHFT);   
    __m512 const INF_VEC = (__m512)_mm512_set1_epi32(INF);
    
    __m512 inf_mask = _MM512_CMP_PS(a, EXP_HI_VEC, _CMP_LT_OS);
    __m512 zero_mask = _MM512_CMP_PS(a, EXP_LO_VEC, _CMP_GT_OS);
    __m512 nan_mask = _MM512_CMP_PS(a, a, 4);
    __m512 inf_vec = _MM512_ANDNOT_PS(inf_mask, INF_VEC);
    __m512 nan_vec = _MM512_AND_PS(a, nan_mask); 
    __m512 fix_mask = _MM512_XOR_PS(zero_mask, inf_mask); 

    __m512i dnrm = _mm512_min_epi32(exp, DNRM_THR_VEC);
            dnrm = _mm512_add_epi32(dnrm, DNRM_SHFT_VEC);
            exp = _mm512_max_epi32(exp, DNRM_THR_VEC);
    __m512 res = (__m512)_mm512_add_epi32(exp, (__m512i)zz);
    res = _mm512_fmadd_ps((__m512)dnrm, res, nan_vec);
    res = _MM512_BLENDV_PS(res, inf_vec, fix_mask);

    return res;
    return zz;
}

