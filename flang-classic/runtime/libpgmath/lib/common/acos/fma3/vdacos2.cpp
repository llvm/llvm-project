
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
#include "dacos_defs.h"

extern "C" __m128d __fvd_acos_fma3(__m128d);

__m128d __fvd_acos_fma3(__m128d const a)
{
    __m128i const ABS_MASK  = _mm_set1_epi64x(ABS_MASK_LL);
    __m128d const ZERO      = _mm_set1_pd(0.0);
    __m128d const ONE       = _mm_set1_pd(1.0);
#if defined(__clang__) && defined(TARGET_ARM64)
    __m128d const SGN_MASK  = (__m128d)((long double)_mm_set1_epi64x(SGN_MASK_LL));
#else
    __m128d const SGN_MASK  = (__m128d)_mm_set1_epi64x(SGN_MASK_LL);
#endif
    __m128d const THRESHOLD = _mm_set1_pd(THRESHOLD_D);
    __m128d const PI_HI     = _mm_set1_pd(PI_HI_D);

    __m128d const A0 = _mm_set1_pd(A0_D);
    __m128d const B0 = _mm_set1_pd(B0_D);
    __m128d const C0 = _mm_set1_pd(C0_D);
    __m128d const D0 = _mm_set1_pd(D0_D);
    __m128d const E0 = _mm_set1_pd(E0_D);
    __m128d const F0 = _mm_set1_pd(F0_D);
    __m128d const G0 = _mm_set1_pd(G0_D);
    __m128d const H0 = _mm_set1_pd(H0_D);
    __m128d const I0 = _mm_set1_pd(I0_D);
    __m128d const J0 = _mm_set1_pd(J0_D);
    __m128d const K0 = _mm_set1_pd(K0_D);
    __m128d const L0 = _mm_set1_pd(L0_D);
    __m128d const M0 = _mm_set1_pd(M0_D);
    __m128d const N0 = _mm_set1_pd(N0_D);

    __m128d const A1 = _mm_set1_pd(A1_D);
    __m128d const B1 = _mm_set1_pd(B1_D);
    __m128d const C1 = _mm_set1_pd(C1_D);
    __m128d const D1 = _mm_set1_pd(D1_D);
    __m128d const E1 = _mm_set1_pd(E1_D);
    __m128d const F1 = _mm_set1_pd(F1_D);
    __m128d const G1 = _mm_set1_pd(G1_D);
    __m128d const H1 = _mm_set1_pd(H1_D);
    __m128d const I1 = _mm_set1_pd(I1_D);
    __m128d const J1 = _mm_set1_pd(J1_D);
    __m128d const K1 = _mm_set1_pd(K1_D);
    __m128d const L1 = _mm_set1_pd(L1_D);
    __m128d const M1 = _mm_set1_pd(M1_D);

    __m128d x, x2, a3, x6, x12, a15, c;
    __m128d sq, p0, p1;
    __m128d res, cmp, sign, fix;
    __m128d p0hi, p0lo, p1hi, p1lo;

#if defined(__clang__) && defined(TARGET_ARM64)
    x  = _mm_and_pd(a, (__m128d)((long double)ABS_MASK));
#else
    x  = _mm_and_pd(a, (__m128d)ABS_MASK);
#endif
    x2 = _mm_mul_pd(a, a);
    sq = _mm_sub_pd(ONE, x);
    sq = _mm_sqrt_pd(sq);

    p1hi = _mm_fmadd_pd(A1, x, B1);

    p1hi = _mm_fmadd_pd(p1hi, x, C1);
    p1lo = _mm_fmadd_pd(H1, x, I1);

    p1hi = _mm_fmadd_pd(p1hi, x, D1);
    p1lo = _mm_fmadd_pd(p1lo, x, J1);
    p0hi = _mm_fmadd_pd(A0, x2, B0);
    p0lo = _mm_fmadd_pd(H0, x2, I0);

    p1hi = _mm_fmadd_pd(p1hi, x, E1);
    p1lo = _mm_fmadd_pd(p1lo, x, K1);
    p0hi = _mm_fmadd_pd(p0hi, x2, C0);
    p0lo = _mm_fmadd_pd(p0lo, x2, J0);

    a3 = _mm_mul_pd(x2, a);
    p1hi = _mm_fmadd_pd(p1hi, x, F1);
    p1lo = _mm_fmadd_pd(p1lo, x, L1);
    p0hi = _mm_fmadd_pd(p0hi, x2, D0);
    p0lo = _mm_fmadd_pd(p0lo, x2, K0);

    p1hi = _mm_fmadd_pd(p1hi, x, G1);
    x6 = _mm_mul_pd(a3, a3);
    p1lo = _mm_fmadd_pd(p1lo, x, M1);
    __m128d pi_mask = _mm_cmp_pd(ZERO, a, _CMP_GT_OQ);
    fix = _mm_cmp_pd(a, ONE, _CMP_GT_OQ);
    p0hi = _mm_fmadd_pd(p0hi, x2, E0);
    p0lo = _mm_fmadd_pd(p0lo, x2, L0);

    p1 = _mm_fmadd_pd(p1hi, x6, p1lo);
    __m128d pi_hi = _mm_and_pd(pi_mask, PI_HI);
    fix = _mm_and_pd(fix, SGN_MASK);
    sign = _mm_and_pd(a, SGN_MASK);
    p0hi = _mm_fmadd_pd(p0hi, x2, F0);
    x12 = _mm_mul_pd(x6, x6);
    p0lo = _mm_fmadd_pd(p0lo, x2, M0);
    c = _mm_sub_pd(N0, a);

    p1 = _mm_fmsub_pd(sq, p1, pi_hi);
    fix = _mm_xor_pd(fix, sign);
    p0hi = _mm_fmadd_pd(p0hi, x2, G0);
    a15 = _mm_mul_pd(x12, a3);
    p0lo = _mm_fmadd_pd(p0lo, a3, c);

    p1 = _mm_xor_pd(p1, fix);
    p0 = _mm_fmadd_pd(p0hi, a15, p0lo);
    cmp = _mm_cmp_pd(x, THRESHOLD, _CMP_LT_OQ);

    res = _mm_blendv_pd(p1, p0, cmp);

    return res;
}

