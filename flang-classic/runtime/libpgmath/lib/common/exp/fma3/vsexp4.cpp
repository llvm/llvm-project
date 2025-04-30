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

#include "exp_defs.h"

extern "C" __m128 __fvs_exp_fma3(__m128);
__m128 __pgm_exp_vec128_slowpath(__m128, __m128i, __m128);

__m128 __fvs_exp_fma3(__m128 a)
{
    __m128 const EXP_HI_VEC = _mm_set1_ps(EXP_HI);
    __m128 const EXP_LO_VEC = _mm_set1_ps(EXP_LO);
    __m128 const EXP_PDN_VEC = _mm_set1_ps(EXP_PDN);
    __m128 const FLT2INT_CVT_VEC = _mm_set1_ps(FLT2INT_CVT);
    __m128 const L2E_VEC = _mm_set1_ps(L2E);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128 const SGN_VEC = (__m128)((long double)_mm_set1_epi32(MASK));
#else
    __m128 const SGN_VEC = (__m128)_mm_set1_epi32(MASK);
#endif

    __m128 abs = _mm_and_ps(a, SGN_VEC);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128i sp_mask = _mm_cmpgt_epi32(_mm_castps_si128((__m128i)((long double)abs)), _mm_castps_si128((__m128i)((long double)EXP_PDN_VEC))); // nie-zero dla niedobrych
#else
    __m128i sp_mask = _mm_cmpgt_epi32(_mm_castps_si128(abs), _mm_castps_si128(EXP_PDN_VEC)); // zero dla dobrych
#endif
#if defined(TARGET_LINUX_POWER)
    int sp = _vec_any_nz(sp_mask);
#else
    int sp = _mm_movemask_epi8(sp_mask);
#endif
    __m128 t = _mm_fmadd_ps(a, L2E_VEC, FLT2INT_CVT_VEC);
    __m128 tt = _mm_sub_ps(t, FLT2INT_CVT_VEC);
    __m128 z = _mm_fnmadd_ps(tt, _mm_set1_ps(LN2_0), a);
           z = _mm_fnmadd_ps(tt, _mm_set1_ps(LN2_1), z);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128i exp = _mm_castps_si128((__m128i)((long double)t));
#else
    __m128i exp = _mm_castps_si128(t);
#endif
            exp = _mm_slli_epi32(exp, 23);

    __m128 zz =                 _mm_set1_ps(EXP_C7);
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C6));
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C5));
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C4));
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C3));
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C2));
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C1));
    zz = _mm_fmadd_ps(zz, z, _mm_set1_ps(EXP_C0));
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128 res = (__m128)((long double)_mm_add_epi32(exp, (__m128i)((long double)zz)));
#else
    __m128 res = (__m128)_mm_add_epi32(exp, (__m128i)zz);
#endif

    if (sp)
    {
        res = __pgm_exp_vec128_slowpath(a, exp, zz);       
    }

    return res;
}


//__m128 __pgm_exp_vec128_slowpath(__m128 a, __m128i exp, __m128 zz) __attribute__((noinline));
__m128 __pgm_exp_vec128_slowpath(__m128 a, __m128i exp, __m128 zz) {
    __m128 const EXP_HI_VEC = _mm_set1_ps(EXP_HI);
    __m128 const EXP_LO_VEC = _mm_set1_ps(EXP_LO);
    __m128i const DNRM_THR_VEC = _mm_set1_epi32(DNRM_THR);
    __m128i const EXP_BIAS_VEC = _mm_set1_epi32(EXP_BIAS);
    __m128i const DNRM_SHFT_VEC = _mm_set1_epi32(DNRM_SHFT);   
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128 const INF_VEC = (__m128)((long double)_mm_set1_epi32(INF));
#else
    __m128 const INF_VEC = (__m128)_mm_set1_epi32(INF);
#endif
    __m128 inf_mask = _mm_cmp_ps(a, EXP_HI_VEC, _CMP_LT_OS);
    __m128 zero_mask = _mm_cmp_ps(a, EXP_LO_VEC, _CMP_GT_OS);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128 nan_mask = (__m128)((long double)_mm_cmp_ps((__m128i)((long double)a), (__m128i)((long double)a), _CMP_NEQ_UQ));
#else
    __m128 nan_mask = _mm_cmp_ps(a, a, _CMP_NEQ_UQ);
#endif
    //ORIG __m128 nan_mask = _mm_cmp_ps(a, a, 4);
    __m128 inf_vec = _mm_andnot_ps(inf_mask, INF_VEC);
    __m128 nan_vec = _mm_and_ps(a, nan_mask); 
    __m128 fix_mask = _mm_xor_ps(zero_mask, inf_mask); 

    __m128i dnrm = _mm_min_epi32(exp, DNRM_THR_VEC);
            dnrm = _mm_add_epi32(dnrm, DNRM_SHFT_VEC);
            exp = _mm_max_epi32(exp, DNRM_THR_VEC);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128 res = (__m128)((long double)_mm_add_epi32(exp, (__m128i)((long double)zz)));
    res = _mm_fmadd_ps((__m128)((long double)dnrm), res, nan_vec);
#else
    __m128 res = (__m128)_mm_add_epi32(exp, (__m128i)zz);
    res = _mm_fmadd_ps((__m128)dnrm, res, nan_vec);
#endif
    res = _mm_blendv_ps(res, inf_vec, fix_mask);

    return res;
}
