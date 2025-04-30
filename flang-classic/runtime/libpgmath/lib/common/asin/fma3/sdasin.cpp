
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
#include <math.h>
#include "dasin_defs.h"

#define FMA __builtin_fma

extern "C" double __fsd_asin_fma3(double);

static long long __double_as_ll(double const a) {
    return *(long long*)&a;
}

static double __ll_as_double(long long const a) {
    return *(double*)&a;
}


double __fsd_asin_fma3(double const a)
{
    __m128d const AH0 = _mm_setr_pd(A0_D, H0_D);
    __m128d const BI0 = _mm_setr_pd(B0_D, I0_D);
    __m128d const CJ0 = _mm_setr_pd(C0_D, J0_D);
    __m128d const DK0 = _mm_setr_pd(D0_D, K0_D);
    __m128d const EL0 = _mm_setr_pd(E0_D, L0_D);
    __m128d const FM0 = _mm_setr_pd(F0_D, M0_D);
    __m128d const Ga0 = _mm_setr_pd(G0_D,    a);

    __m128d const AG1 = _mm_setr_pd(A1_D, G1_D);
    __m128d const BH1 = _mm_setr_pd(B1_D, H1_D);
    __m128d const CI1 = _mm_setr_pd(C1_D, I1_D);
    __m128d const DJ1 = _mm_setr_pd(D1_D, J1_D);
    __m128d const EK1 = _mm_setr_pd(E1_D, K1_D);
    __m128d const FL1 = _mm_setr_pd(F1_D, L1_D);

    double res;
    double x = __ll_as_double(ABS_MASK_LL & __double_as_ll(a));
    double x2 = a * a;
    double a3 = x2 * a;
    double x6 = a3 * a3;

    if (__double_as_ll(x) < THRESHOLD_LL)
    {
        __m128d _x2 = _mm_set1_pd(x2);
        __m128d _x2_a3 = _mm_setr_pd(x2, a3);
        __m128d _p0;

        _p0 = _mm_fmadd_pd(AH0, _x2, BI0);
        _p0 = _mm_fmadd_pd(_p0, _x2, CJ0);
        _p0 = _mm_fmadd_pd(_p0, _x2, DK0);
        _p0 = _mm_fmadd_pd(_p0, _x2, EL0);
        _p0 = _mm_fmadd_pd(_p0, _x2, FM0);

        _p0 = _mm_fmadd_pd(_p0, _x2_a3, Ga0);

        double p0hi = _mm_cvtsd_f64(_p0);
        _p0 = _mm_shuffle_pd(_p0, _p0, 3);
        double p0lo = _mm_cvtsd_f64(_p0);

        double x12 = x6 * x6;
        double a15 = x12 * a3;
        res = FMA(p0hi, a15, p0lo);
    }
    else
    {
        double sq = 1.0 - x;
        /*
	 * There seems to be a concensus that setting errno is important
	 * for fastmath intrinsics.
	 * Disable using Intel hardware instruction sqrt.
	 */
	sq = sqrt(sq);

        long long fix = (long long)(a > 1.0) << 63;
        long long sign = SGN_MASK_LL & __double_as_ll(a);
        fix = fix ^ sign;

        __m128d _x = _mm_set1_pd(x);
        __m128d _p1;
        _p1 = _mm_fmadd_pd(AG1, _x, BH1);
        _p1 = _mm_fmadd_pd(_p1, _x, CI1);
        _p1 = _mm_fmadd_pd(_p1, _x, DJ1);
        _p1 = _mm_fmadd_pd(_p1, _x, EK1);
        _p1 = _mm_fmadd_pd(_p1, _x, FL1);

        double p1hi = _mm_cvtsd_f64(_p1);
        _p1 = _mm_shuffle_pd(_p1, _p1, 1);
        double p1lo = _mm_cvtsd_f64(_p1);

        double p1 = FMA(p1hi, x6, p1lo);
        p1 = FMA(sq, p1, PIO2_LO_D);
        p1 = p1 + PIO2_HI_D;

        res = __ll_as_double(fix ^ __double_as_ll(p1));
    }


    return res;
}
