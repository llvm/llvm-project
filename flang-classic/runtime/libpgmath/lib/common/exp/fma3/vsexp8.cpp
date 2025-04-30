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
#include "exp_defs.h"

extern "C" __m256 __fvs_exp_fma3_256(__m256);
__m256 __pgm_exp_vec256_slowpath(__m256, __m256i, __m256);

__m256 __fvs_exp_fma3_256(__m256 a)
{
    __m256 const EXP_HI_VEC = _mm256_set1_ps(EXP_HI);
    __m256 const EXP_LO_VEC = _mm256_set1_ps(EXP_LO);
    __m256 const EXP_PDN_VEC = _mm256_set1_ps(EXP_PDN);
    __m256 const FLT2INT_CVT_VEC = _mm256_set1_ps(FLT2INT_CVT);
    __m256 const L2E_VEC = _mm256_set1_ps(L2E);
   
    __m256 const SGN_VEC = (__m256)_mm256_set1_epi32(MASK);

    __m256 abs = _mm256_and_ps(a, SGN_VEC);
    __m256i sp_mask = _mm256_cmpgt_epi32(_mm256_castps_si256(abs), _mm256_castps_si256(EXP_PDN_VEC)); // zero dla dobrych
    int sp = _mm256_movemask_epi8(sp_mask);
    __m256 t = _mm256_fmadd_ps(a, L2E_VEC, FLT2INT_CVT_VEC);
    __m256 tt = _mm256_sub_ps(t, FLT2INT_CVT_VEC);
    __m256 z = _mm256_fnmadd_ps(tt, _mm256_set1_ps(LN2_0), a);
           z = _mm256_fnmadd_ps(tt, _mm256_set1_ps(LN2_1), z);
         
    __m256i exp = _mm256_castps_si256(t);
            exp = _mm256_slli_epi32(exp, 23);

    __m256 zz =                 _mm256_set1_ps(EXP_C7);
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C6));
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C5));
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C4));
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C3));
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C2));
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C1));
    zz = _mm256_fmadd_ps(zz, z, _mm256_set1_ps(EXP_C0));
    __m256 res = (__m256)_mm256_add_epi32(exp, (__m256i)zz);
 
    if (sp)
    {
        res = __pgm_exp_vec256_slowpath(a, exp, zz);       
    }

    return res;
}


__m256 __pgm_exp_vec256_slowpath(__m256 a, __m256i exp, __m256 zz) {
    __m256 const EXP_HI_VEC = _mm256_set1_ps(EXP_HI);
    __m256 const EXP_LO_VEC = _mm256_set1_ps(EXP_LO);
    __m256i const DNRM_THR_VEC = _mm256_set1_epi32(DNRM_THR);
    __m256i const EXP_BIAS_VEC = _mm256_set1_epi32(EXP_BIAS);
    __m256i const DNRM_SHFT_VEC = _mm256_set1_epi32(DNRM_SHFT);   
    __m256 const INF_VEC = (__m256)_mm256_set1_epi32(INF);
    
    __m256 inf_mask = _mm256_cmp_ps(a, EXP_HI_VEC, _CMP_LT_OS);
    __m256 zero_mask = _mm256_cmp_ps(a, EXP_LO_VEC, _CMP_GT_OS);
    __m256 nan_mask = _mm256_cmp_ps(a, a, 4);
    __m256 inf_vec = _mm256_andnot_ps(inf_mask, INF_VEC);
    __m256 nan_vec = _mm256_and_ps(a, nan_mask); 
    __m256 fix_mask = _mm256_xor_ps(zero_mask, inf_mask); 

    __m256i dnrm = _mm256_min_epi32(exp, DNRM_THR_VEC);
            dnrm = _mm256_add_epi32(dnrm, DNRM_SHFT_VEC);
            exp = _mm256_max_epi32(exp, DNRM_THR_VEC);
    __m256 res = (__m256)_mm256_add_epi32(exp, (__m256i)zz);
    res = _mm256_fmadd_ps((__m256)dnrm, res, nan_vec);
    res = _mm256_blendv_ps(res, inf_vec, fix_mask);

    return res;
    return zz;
}

