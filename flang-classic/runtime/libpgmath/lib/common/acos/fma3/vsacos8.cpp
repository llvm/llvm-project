
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

extern "C" __m256 __fvs_acos_fma3_256(__m256 const a);

__m256 __fvs_acos_fma3_256(__m256 const a)
{
    __m256  const ABS_MASK      = (__m256)_mm256_set1_epi32(ABS_MASK_I);
    __m256  const SGN_MASK      = (__m256)_mm256_set1_epi32(SGN_MASK_I);
    __m256  const ONE           = _mm256_set1_ps(1.0f);
    __m256i const ZERO          = _mm256_set1_epi32(0);
    __m256i const THRESHOLD     = (__m256i)_mm256_set1_ps(THRESHOLD_F);
    __m256  const PI            = _mm256_set1_ps(PI_F);

    // p0 coefficients
    __m256 const A0 = _mm256_set1_ps(A0_F);
    __m256 const B0 = _mm256_set1_ps(B0_F);
    __m256 const C0 = _mm256_set1_ps(C0_F);
    __m256 const D0 = _mm256_set1_ps(D0_F);
    __m256 const E0 = _mm256_set1_ps(E0_F);
    __m256 const F0 = _mm256_set1_ps(F0_F);

    // p1 coefficients
    __m256 const A1 = _mm256_set1_ps(A1_F);
    __m256 const B1 = _mm256_set1_ps(B1_F);
    __m256 const C1 = _mm256_set1_ps(C1_F);
    __m256 const D1 = _mm256_set1_ps(D1_F);
    __m256 const E1 = _mm256_set1_ps(E1_F);
    __m256 const F1 = _mm256_set1_ps(F1_F);

    __m256 x, x2, a3, sq, p0, p1, res, c, cmp0;
    x = _mm256_and_ps(ABS_MASK, a);
    sq = _mm256_sub_ps(ONE, x);
    sq = _mm256_sqrt_ps(sq); // sqrt(1 - |a|)

    __m256 pi_mask = (__m256)_mm256_cmpgt_epi32(ZERO, (__m256i)a);
    cmp0 = (__m256)_mm256_cmpgt_epi32((__m256i)x, THRESHOLD);

    // polynomials evaluation
    x2 = _mm256_mul_ps(a, a);
    c  = _mm256_sub_ps(F0, a);
    p1 = _mm256_fmadd_ps(A1, x, B1);
    p0 = _mm256_fmadd_ps(A0, x2, B0);
    p1 = _mm256_fmadd_ps(p1, x, C1);
    p0 = _mm256_fmadd_ps(p0, x2, C0);
    p1 = _mm256_fmadd_ps(p1, x, D1);
    a3 = _mm256_mul_ps(x2, a);
    p0 = _mm256_fmadd_ps(p0, x2, D0);
    p1 = _mm256_fmadd_ps(p1, x, E1);
    p0 = _mm256_fmadd_ps(p0, x2, E0);
    p1 = _mm256_fmadd_ps(p1, x, F1);
    p0 = _mm256_fmadd_ps(p0, a3, c);

    pi_mask = _mm256_and_ps(pi_mask, PI);
    p1 = _mm256_fmsub_ps(sq, p1, pi_mask);

    __m256 sign;
    sign = _mm256_and_ps(a, SGN_MASK);

    __m256 fix;
    fix = _mm256_cmp_ps(a, ONE, _CMP_GT_OQ);
    fix = _mm256_and_ps(fix, SGN_MASK);
    fix = _mm256_xor_ps(fix, sign);
    p1 = _mm256_xor_ps(p1, fix);

    res = _mm256_blendv_ps(p0, p1, cmp0);

    return res;
}
