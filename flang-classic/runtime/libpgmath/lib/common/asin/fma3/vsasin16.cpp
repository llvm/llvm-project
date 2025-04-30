
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
#include "asin_defs.h"

//#define INTEL_NAN

extern "C" __m512 FCN_AVX512(__fvs_asin_fma3)(__m512 const a);

__m512 FCN_AVX512(__fvs_asin_fma3)(__m512 const a) {
    __m512  const ABS_MASK  = (__m512)_mm512_set1_epi32(ABS_MASK_I);
    __m512  const SGN_MASK  = (__m512)_mm512_set1_epi32(SGN_MASK_I);
    __m512  const ONE       = _mm512_set1_ps(ONE_F);
    __m512i const THRESHOLD = (__m512i)_mm512_set1_ps(THRESHOLD_F);
    __m512  const PIO2      = _mm512_set1_ps(PIO2_F);

    // p0 coefficients
    __m512 const A0 = _mm512_set1_ps(A0_F);
    __m512 const B0 = _mm512_set1_ps(B0_F);
    __m512 const C0 = _mm512_set1_ps(C0_F);
    __m512 const D0 = _mm512_set1_ps(D0_F);
    __m512 const E0 = _mm512_set1_ps(E0_F);

    // p1 coefficients
    __m512 const A1 = _mm512_set1_ps(A1_F);
    __m512 const B1 = _mm512_set1_ps(B1_F);
    __m512 const C1 = _mm512_set1_ps(C1_F);
    __m512 const D1 = _mm512_set1_ps(D1_F);
    __m512 const E1 = _mm512_set1_ps(E1_F);
    __m512 const F1 = _mm512_set1_ps(F1_F);
    __m512 const G1 = _mm512_set1_ps(G1_F);

    __m512 x, x2, x3, sq, p0, p1, res, cmp0;

    x = _MM512_AND_PS(ABS_MASK, a);
    sq = _mm512_sub_ps(ONE, x);
    sq = _mm512_sqrt_ps(sq); // sqrt(1 - |a|)

    // sgn(a) * ( |a| > 0.5705 ? pi/2 - sqrt(1 - |x|) * p1(|a|) : p0(|a|) )
    cmp0 = (__m512)_MM512_CMPGT_EPI32((__m512i)x, THRESHOLD);

    // polynomials evaluation
    x2 = _mm512_mul_ps(a, a);
    p1 = _mm512_fmadd_ps(A1, x, B1);
    p0 = _mm512_fmadd_ps(A0, x2, B0);
    p1 = _mm512_fmadd_ps(p1, x, C1);
    p0 = _mm512_fmadd_ps(p0, x2, C0);
    p1 = _mm512_fmadd_ps(p1, x, D1);
    x3 = _mm512_mul_ps(x2, x);
    p0 = _mm512_fmadd_ps(p0, x2, D0);
    p1 = _mm512_fmadd_ps(p1, x, E1);
    p0 = _mm512_fmadd_ps(p0, x2, E0);
    p1 = _mm512_fmadd_ps(p1, x, F1);
    p1 = _mm512_fmadd_ps(p1, x, G1);
    p0 = _mm512_fmadd_ps(p0, x3, x);

    p1 = _mm512_fmadd_ps(sq, p1, PIO2);
    res = _MM512_BLENDV_PS(p0, p1, cmp0);

#ifndef INTEL_NAN // GCC NAN:
    __m512 sign, fix;
    sign = _MM512_AND_PS(a, SGN_MASK);
    fix = _MM512_CMP_PS(a, ONE, _CMP_GT_OQ);
    fix = _MM512_AND_PS(fix, SGN_MASK);
    fix = _MM512_XOR_PS(fix, sign);
    res = _MM512_XOR_PS(res, fix);
#else // INTEL NAN:
    __m512 sign;
    sign = _MM512_AND_PS(a, SGN_MASK);
    res = _MM512_OR_PS(res, sign);
#endif

    return res;
}
