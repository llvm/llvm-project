
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
#include "dacos_defs.h"

#define FMA __builtin_fma

extern "C" double __fsd_acos_fma3(double);

static long long __double_as_ll(double const a) {
    return *(long long*)&a;
}

static double __ll_as_double(long long const a) {
    return *(double*)&a;
}

double __fsd_acos_fma3(double const a)
{
    __m128d const AH0 = _mm_setr_pd(A0_D, H0_D);
    __m128d const BI0 = _mm_setr_pd(B0_D, I0_D);
    __m128d const CJ0 = _mm_setr_pd(C0_D, J0_D);
    __m128d const DK0 = _mm_setr_pd(D0_D, K0_D);
    __m128d const EL0 = _mm_setr_pd(E0_D, L0_D);
    __m128d const FM0 = _mm_setr_pd(F0_D, M0_D);
    __m128d const GN0 = _mm_setr_pd(G0_D, N0_D - a);

    __m128d const AH1 = _mm_setr_pd(A1_D, H1_D);
    __m128d const BI1 = _mm_setr_pd(B1_D, I1_D);
    __m128d const CJ1 = _mm_setr_pd(C1_D, J1_D);
    __m128d const DK1 = _mm_setr_pd(D1_D, K1_D);
    __m128d const EL1 = _mm_setr_pd(E1_D, L1_D);
    __m128d const FM1 = _mm_setr_pd(F1_D, M1_D);

    double res;
    double x = __ll_as_double(ABS_MASK_LL & __double_as_ll(a));
    double x2 = a * a;
    double a3 = x2 * a;
    double x6 = a3 * a3;

    if (__double_as_ll(x) >= __double_as_ll(THRESHOLD_D))
    {
        double sq = 1.0 - x;
	/*
	 * There seems to be a concensus that setting errno is important
	 * for fastmath intrinsics.
	 * Disable using Intel hardware instruction sqrt.
	 */
        sq = sqrt(sq);

        double pi_hi = a < 0.0 ? PI_HI_D : 0.0;
        long long fix = (long long)(a > 1.0) << 63;
        long long sign = SGN_MASK_LL & __double_as_ll(a);
        fix = fix ^ sign;

        __m128d _x = _mm_set1_pd(x);
        __m128d _p1;
        _p1 = _mm_fmadd_pd(AH1, _x, BI1);
        _p1 = _mm_fmadd_pd(_p1, _x, CJ1);
        _p1 = _mm_fmadd_pd(_p1, _x, DK1);
        _p1 = _mm_fmadd_pd(_p1, _x, EL1);
        _p1 = _mm_fmadd_pd(_p1, _x, FM1);

        double p1hi = _mm_cvtsd_f64(_p1);
        _p1 = _mm_shuffle_pd(_p1, _p1, 3);
        double p1lo = _mm_cvtsd_f64(_p1);

        p1hi = FMA(p1hi, x, G1_D);
        double p1 = FMA(p1hi, x6, p1lo);
        p1 = FMA(sq, p1, -pi_hi);

        res = __ll_as_double(fix ^ __double_as_ll(p1));
    }
    else
    {
        __m128d _x2 = _mm_set1_pd(x2);
        __m128d _x2_a3 = _mm_setr_pd(x2, a3);
        __m128d _p0;

        _p0 = _mm_fmadd_pd(AH0, _x2, BI0);
        _p0 = _mm_fmadd_pd(_p0, _x2, CJ0);
        _p0 = _mm_fmadd_pd(_p0, _x2, DK0);
        _p0 = _mm_fmadd_pd(_p0, _x2, EL0);
        _p0 = _mm_fmadd_pd(_p0, _x2, FM0);
        _p0 = _mm_fmadd_pd(_p0, _x2_a3, GN0);

        double p0hi = _mm_cvtsd_f64(_p0);
        _p0 = _mm_shuffle_pd(_p0, _p0, 3);
        double p0lo = _mm_cvtsd_f64(_p0);

        double x12 = x6 * x6;
        double a15 = x12 * a3;
        res = FMA(p0hi, a15, p0lo);
    }

    return res;
}


