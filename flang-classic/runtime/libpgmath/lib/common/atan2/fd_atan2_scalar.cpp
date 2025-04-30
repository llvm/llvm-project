
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>
#include <math.h>

#if !(defined _CPU)
#error: please define _CPU - specific suffix to a function name
#endif

#define _JOIN2(a,b) a##b
#define JOIN2(a,b) _JOIN2(a,b)

#define atan2_d_scalar JOIN2(__fd_atan2_1_,_CPU)
#define FMA __builtin_fma

extern "C" double atan2_d_scalar(double,double);


static unsigned long long int __attribute__ ((always_inline)) as_ulong(double x)
{
    return *(unsigned long long int *)&x;
}

static double __attribute__ ((always_inline)) as_double(unsigned long long int x)
{ 
    return *(double *)&x;
}

// We use the relationship between atan2(y,x) and atan(x),
// as described in:
// https://en.wikipedia.org/wiki/Atan2
// to create an atan2(y, x) implementation based on our atan(x) implementation
// Namely:
// atan2(y, x) = atan(y/x) for x > 0
// atan2(y, x) = atan(y/x) + pi for x < 0 and y >= 0
// atan2(y, x) = atan(y/x) - pi for x < 0 and y < 0
// atan2(y, x) =  pi/2 for x = 0 and y > 0
// atan2(y, x) = -pi/2 for x = 0 and y < 0

// Also, from the C99 standards we need:
// atan2(+-0.0, +0.0) = +-0.0
// atan2(+-0.0, -0.0) = +-PI

// Special care need to be taken when both x and y are +-INFINITY, where
// the ieee definitions are equivalent to letting x and y tend to infinity at
// the same rate.

double atan2_d_scalar(double const y, double const x) {

    // Special return values when both inputs are infinity, or 0's
    // (or any absolute equal numbers, the results are the same):
    if (__builtin_expect(fabs(y) == fabs(x), 1)) {

        // Using (as_ulong(x) > 0x7FFFFFFFFFFFFFFF) rather than (x < 0.0) here
        // seems to give a performance boost when looping over this function
        // many times
        double ans =
            FMA((as_ulong(x) > 0x7FFFFFFFFFFFFFFF), PI_OVER_2, PI_OVER_4);

        // Special return values for (y, x) = (+-0.0, +-0.0)
        if (x == 0.0) {
            ans = (as_ulong(x) == 0x0) ? 0.0 : PI;
        }

        return copysign(ans, y);
    }

    double xReduced;

    // xReduced = ((fabs(y) > fabs(x)) ? x : y) / ((fabs(y) > fabs(x)) ? y : x);
    // Seems to be the fastest way of getting x/y or y/x:
    if (fabs(y) > fabs(x)) {
        xReduced = x / y;
    } else {
        xReduced = y / x;
    }

    // The same Estrin scheme as is used in atan(x):
    double x2 = xReduced * xReduced;
    double x4 = x2 * x2;
    double x8 = x4 * x4;
    double x16 = x8 * x8;

    double L1 = FMA(x2, C6, C5);
    double L2 = FMA(x2, C8, C7);
    double L3 = FMA(x2, C10, C9);
    double L4 = FMA(x2, C12, C11);
    double L5 = FMA(x2, C14, C13);
    double L6 = FMA(x2, C16, C15);
    double L7 = FMA(x2, C18, C17);
    double L8 = FMA(x2, C20, C19);

    // L1 + x4*L2 + x8*L3 + x12*L4 + x16*L5 + x20*L6 + x24*L7 + x28*L8
    double M1 = FMA(x4, L2, L1);
    double M2 = FMA(x4, L4, L3);
    double M3 = FMA(x4, L6, L5);
    double M4 = FMA(x4, L8, L7);

    // M1 + x8*M2 + x16*M3 + x24*M4
    // (M1 + x8*M2) + x16*(M3 + x8*M4)
    // (M1 + x8*M2) + x16*(M3 + x8*M4)
    double N1 = FMA(x8, M2, M1);
    double N2 = FMA(x8, M4, M3);

    // c2 + x2*c3 + x4*c4 + x6*(N1 + x16*N2):
    double poly = FMA(x16, N2, N1);

    poly = FMA(x4, FMA(x2, poly, C4), FMA(x2, C3, C2));

    double result_d = poly;

    double pi_factor = 0.0;

    if (fabs(y) > fabs(x)) {
        // pi/2 with the sign of xReduced:
        const double signedPi_2 = as_double(
            as_ulong(PI_OVER_2) | (as_ulong(xReduced) & 0x8000000000000000));

        xReduced = -xReduced;
        pi_factor = signedPi_2;
    }

    result_d = FMA(x2 * xReduced, poly, xReduced);

    // Faster than the if statement:
    // pi with the sign of y:
    const double signedPi =
        as_double(as_ulong(PI) | (as_ulong(y) & 0x8000000000000000));

    // Again, faster:
    pi_factor = FMA((as_ulong(x) > 0x7FFFFFFFFFFFFFFF), signedPi, pi_factor);
    // if ((as_ulong(x) > 0x7FFFFFFFFFFFFFFF)) {
    //    pi_factor += signedPi;
    //}

    result_d += pi_factor;

    // Unfortunately we have to do a copysign here to return the correctly
    // signed 0.0 for atan2(+-0.0, x > 0.0) inputs:
    result_d = copysign(result_d, y);

#ifdef IACA
#pragma message("IACA END")
    IACA_END
#endif

    return result_d;
}
