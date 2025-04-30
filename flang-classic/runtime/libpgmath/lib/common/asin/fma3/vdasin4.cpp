
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
#include "dasin_defs.h"

extern "C" __m256d __fvd_asin_fma3_256(__m256d);

__m256d __fvd_asin_fma3_256(__m256d const a)
{
    __m256i const ABS_MASK  = _mm256_set1_epi64x(ABS_MASK_LL);
    __m256d const ZERO      = _mm256_set1_pd(0.0);
    __m256d const ONE       = _mm256_set1_pd(1.0);
    __m256d const SGN_MASK  = (__m256d)_mm256_set1_epi64x(SGN_MASK_LL);
    __m256d const THRESHOLD = (__m256d)_mm256_set1_epi64x(THRESHOLD_LL);
    __m256d const PIO2_HI   = _mm256_set1_pd(PIO2_HI_D);
    __m256d const PIO2_LO   = _mm256_set1_pd(PIO2_LO_D);

    __m256d const A0 = _mm256_set1_pd(A0_D);
    __m256d const B0 = _mm256_set1_pd(B0_D);
    __m256d const C0 = _mm256_set1_pd(C0_D);
    __m256d const D0 = _mm256_set1_pd(D0_D);
    __m256d const E0 = _mm256_set1_pd(E0_D);
    __m256d const F0 = _mm256_set1_pd(F0_D);
    __m256d const G0 = _mm256_set1_pd(G0_D);
    __m256d const H0 = _mm256_set1_pd(H0_D);
    __m256d const I0 = _mm256_set1_pd(I0_D);
    __m256d const J0 = _mm256_set1_pd(J0_D);
    __m256d const K0 = _mm256_set1_pd(K0_D);
    __m256d const L0 = _mm256_set1_pd(L0_D);
    __m256d const M0 = _mm256_set1_pd(M0_D);

    __m256d const A1 = _mm256_set1_pd(A1_D);
    __m256d const B1 = _mm256_set1_pd(B1_D);
    __m256d const C1 = _mm256_set1_pd(C1_D);
    __m256d const D1 = _mm256_set1_pd(D1_D);
    __m256d const E1 = _mm256_set1_pd(E1_D);
    __m256d const F1 = _mm256_set1_pd(F1_D);
    __m256d const G1 = _mm256_set1_pd(G1_D);
    __m256d const H1 = _mm256_set1_pd(H1_D);
    __m256d const I1 = _mm256_set1_pd(I1_D);
    __m256d const J1 = _mm256_set1_pd(J1_D);
    __m256d const K1 = _mm256_set1_pd(K1_D);
    __m256d const L1 = _mm256_set1_pd(L1_D);

    __m256d x, x2, x3, x6, x12, x15;
    __m256d sq, p0hi, p0lo, p0, p1hi, p1lo, p1;
    __m256d res, cmp, sign, fix, pio2_lo, pio2_hi;

    x  = _mm256_and_pd(a, (__m256d)ABS_MASK);
    sq = _mm256_sub_pd(ONE, x);
    sq = _mm256_sqrt_pd(sq);

    x2 = _mm256_mul_pd(a, a);
    p1hi = _mm256_fmadd_pd(A1, x, B1);
    p1lo = _mm256_fmadd_pd(G1, x, H1);

    p0hi = _mm256_fmadd_pd(A0, x2, B0);
    p0lo = _mm256_fmadd_pd(H0, x2, I0);
    p1hi = _mm256_fmadd_pd(p1hi, x, C1);
    p1lo = _mm256_fmadd_pd(p1lo, x, I1);

    p0hi = _mm256_fmadd_pd(p0hi, x2, C0);
    p0lo = _mm256_fmadd_pd(p0lo, x2, J0);
    p1hi = _mm256_fmadd_pd(p1hi, x, D1);
    p1lo = _mm256_fmadd_pd(p1lo, x, J1);

    p0hi = _mm256_fmadd_pd(p0hi, x2, D0);
    p0lo = _mm256_fmadd_pd(p0lo, x2, K0);
    fix = _mm256_cmp_pd(a, ONE, _CMP_GT_OQ);
    x3 = _mm256_mul_pd(x2, x);
    p1hi = _mm256_fmadd_pd(p1hi, x, E1);
    p1lo = _mm256_fmadd_pd(p1lo, x, K1);

    p0hi = _mm256_fmadd_pd(p0hi, x2, E0);
    p0lo = _mm256_fmadd_pd(p0lo, x2, L0);
    sign = _mm256_and_pd(a, SGN_MASK);
    fix = _mm256_and_pd(fix, SGN_MASK);
    x6 = _mm256_mul_pd(x3, x3);
    p1hi = _mm256_fmadd_pd(p1hi, x, F1);
    p1lo = _mm256_fmadd_pd(p1lo, x, L1);

    x12 = _mm256_mul_pd(x6, x6);
    p0hi = _mm256_fmadd_pd(p0hi, x2, F0);
    p0lo = _mm256_fmadd_pd(p0lo, x2, M0);
    fix = _mm256_xor_pd(fix, sign);
    p1 = _mm256_fmadd_pd(p1hi, x6, p1lo);

    x15 = _mm256_mul_pd(x12, x3);
    p0hi = _mm256_fmadd_pd(p0hi, x2, G0);
    p0lo = _mm256_fmadd_pd(p0lo, x3, x);

    p0 = _mm256_fmadd_pd(p0hi, x15, p0lo);
    p1 = _mm256_fmadd_pd(sq, p1, PIO2_LO);

    cmp = _mm256_cmp_pd(x, THRESHOLD, _CMP_LT_OQ);
    p1 = _mm256_add_pd(p1, PIO2_HI);

    res = _mm256_blendv_pd(p1, p0, cmp);
    res = _mm256_xor_pd(res, fix);

    return res;
}
