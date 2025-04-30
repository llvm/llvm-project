
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
#include "acos_defs.h"
#include "mth_avx512helper.h"

extern "C" __m512 FCN_AVX512(__fvs_acos_fma3)(__m512 const a);

__m512 FCN_AVX512(__fvs_acos_fma3)(__m512 const a)
{
    __m512  const ABS_MASK      = (__m512)_mm512_set1_epi32(ABS_MASK_I);
    __m512  const SGN_MASK      = (__m512)_mm512_set1_epi32(SGN_MASK_I);
    __m512  const ONE           = _mm512_set1_ps(1.0f);
    __m512i const ZERO          = _mm512_set1_epi32(0);
    __m512i const THRESHOLD     = (__m512i)_mm512_set1_ps(THRESHOLD_F);
    __m512  const PI            = _mm512_set1_ps(PI_F);

    // p0 coefficients
    __m512 const A0 = _mm512_set1_ps(A0_F);
    __m512 const B0 = _mm512_set1_ps(B0_F);
    __m512 const C0 = _mm512_set1_ps(C0_F);
    __m512 const D0 = _mm512_set1_ps(D0_F);
    __m512 const E0 = _mm512_set1_ps(E0_F);
    __m512 const F0 = _mm512_set1_ps(F0_F);

    // p1 coefficients
    __m512 const A1 = _mm512_set1_ps(A1_F);
    __m512 const B1 = _mm512_set1_ps(B1_F);
    __m512 const C1 = _mm512_set1_ps(C1_F);
    __m512 const D1 = _mm512_set1_ps(D1_F);
    __m512 const E1 = _mm512_set1_ps(E1_F);
    __m512 const F1 = _mm512_set1_ps(F1_F);

    __m512 x, x2, a3, sq, p0, p1, res, c, cmp0;
    x = _MM512_AND_PS(ABS_MASK, a);
    sq = _mm512_sub_ps(ONE, x);
    sq = _mm512_sqrt_ps(sq); // sqrt(1 - |a|)

    __m512 pi_mask = (__m512)_MM512_CMPGT_EPI32(ZERO, (__m512i)a);
    cmp0 = (__m512)_MM512_CMPGT_EPI32((__m512i)x, THRESHOLD);

    // polynomials evaluation
    x2 = _mm512_mul_ps(a, a);
    c  = _mm512_sub_ps(F0, a);
    p1 = _mm512_fmadd_ps(A1, x, B1);
    p0 = _mm512_fmadd_ps(A0, x2, B0);
    p1 = _mm512_fmadd_ps(p1, x, C1);
    p0 = _mm512_fmadd_ps(p0, x2, C0);
    p1 = _mm512_fmadd_ps(p1, x, D1);
    a3 = _mm512_mul_ps(x2, a);
    p0 = _mm512_fmadd_ps(p0, x2, D0);
    p1 = _mm512_fmadd_ps(p1, x, E1);
    p0 = _mm512_fmadd_ps(p0, x2, E0);
    p1 = _mm512_fmadd_ps(p1, x, F1);
    p0 = _mm512_fmadd_ps(p0, a3, c);

    pi_mask = _MM512_AND_PS(pi_mask, PI);
    p1 = _mm512_fmsub_ps(sq, p1, pi_mask);

    __m512 sign;
    sign = _MM512_AND_PS(a, SGN_MASK);

    __m512 fix;
    fix = _MM512_CMP_PS(a, ONE, _CMP_GT_OQ);
    fix = _MM512_AND_PS(fix, SGN_MASK);
    fix = _MM512_XOR_PS(fix, sign);
    p1 = _MM512_XOR_PS(p1, fix);

    res = _MM512_BLENDV_PS(p0, p1, cmp0);

    return res;
}
