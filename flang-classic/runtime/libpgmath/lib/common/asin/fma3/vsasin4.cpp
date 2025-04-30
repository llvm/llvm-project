
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
#include "asin_defs.h"

//#define INTEL_NAN

extern "C" __m128 __fvs_asin_fma3(__m128 const a);

__m128 __fvs_asin_fma3(__m128 const a) {
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128  const ABS_MASK  = (__m128)((long double)_mm_set1_epi32(ABS_MASK_I));
    __m128  const SGN_MASK  = (__m128)((long double)_mm_set1_epi32(SGN_MASK_I));
#else
    __m128  const ABS_MASK  = (__m128)_mm_set1_epi32(ABS_MASK_I);
    __m128  const SGN_MASK  = (__m128)_mm_set1_epi32(SGN_MASK_I);
#endif
    __m128  const ONE       = _mm_set1_ps(1.0f);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128i const THRESHOLD = (__m128i)((long double)_mm_set1_ps(THRESHOLD_F));
#else
    __m128i const THRESHOLD = (__m128i)_mm_set1_ps(THRESHOLD_F);
#endif
    __m128  const PIO2      = _mm_set1_ps(PIO2_F);

    // p0 coefficients
    __m128 const A0 = _mm_set1_ps(A0_F);
    __m128 const B0 = _mm_set1_ps(B0_F);
    __m128 const C0 = _mm_set1_ps(C0_F);
    __m128 const D0 = _mm_set1_ps(D0_F);
    __m128 const E0 = _mm_set1_ps(E0_F);

    // p1 coefficients
    __m128 const A1 = _mm_set1_ps(A1_F);
    __m128 const B1 = _mm_set1_ps(B1_F);
    __m128 const C1 = _mm_set1_ps(C1_F);
    __m128 const D1 = _mm_set1_ps(D1_F);
    __m128 const E1 = _mm_set1_ps(E1_F);
    __m128 const F1 = _mm_set1_ps(F1_F);
    __m128 const G1 = _mm_set1_ps(G1_F);

    __m128 x, x2, x3, sq, p0, p1, res, cmp0;
    x = _mm_and_ps(ABS_MASK, a);
    sq = _mm_sub_ps(ONE, x);
    sq = _mm_sqrt_ps(sq); // sqrt(1 - |a|)

    // sgn(a) * ( |a| > 0.5705 ? pi/2 - sqrt(1 - |x|) * p1(|a|) : p0(|a|) )
#if defined(__clang__) && defined(TARGET_ARM64)
    cmp0 = (__m128)((long double)_mm_cmpgt_epi32((__m128i)((long double)x), THRESHOLD));
#else
    cmp0 = (__m128)_mm_cmpgt_epi32((__m128i)x, THRESHOLD);
#endif

    // polynomials evaluation
    x2 = _mm_mul_ps(a, a);
    p1 = _mm_fmadd_ps(A1, x, B1);
    p0 = _mm_fmadd_ps(A0, x2, B0);
    p1 = _mm_fmadd_ps(p1, x, C1);
    p0 = _mm_fmadd_ps(p0, x2, C0);
    p1 = _mm_fmadd_ps(p1, x, D1);
    x3 = _mm_mul_ps(x2, x);
    p0 = _mm_fmadd_ps(p0, x2, D0);
    p1 = _mm_fmadd_ps(p1, x, E1);
    p0 = _mm_fmadd_ps(p0, x2, E0);
    p1 = _mm_fmadd_ps(p1, x, F1);
    p1 = _mm_fmadd_ps(p1, x, G1);
    p0 = _mm_fmadd_ps(p0, x3, x);

    p1 = _mm_fmadd_ps(sq, p1, PIO2);
    res = _mm_blendv_ps(p0, p1, cmp0);

#ifndef INTEL_NAN // GCC NAN:
    __m128 sign, fix;
    sign = _mm_and_ps(a, SGN_MASK);
    fix = _mm_cmp_ps(a, ONE, _CMP_GT_OQ);
    fix = _mm_and_ps(fix, SGN_MASK);
    fix = _mm_xor_ps(fix, sign);
    res = _mm_xor_ps(res, fix);
#else // INTEL NAN:
    __m128 sign;
    sign = _mm_and_ps(a, SGN_MASK);
    res = _mm_or_ps(res, sign);
#endif

    return res;
}
