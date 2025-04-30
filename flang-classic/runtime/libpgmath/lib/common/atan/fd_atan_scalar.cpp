
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>
#include <immintrin.h>
#include <math.h>

// unsigned long int as_ulong(double x){
//     return *(unsigned long int*)&x;
// }

#define _JOIN2(a,b) a##b
#define JOIN2(a,b) _JOIN2(a,b)

#define atan_d_scalar JOIN2(__fd_atan_1_,_CPU)
#define FMA __builtin_fma

extern "C" double atan_d_scalar(double);

double __attribute__((noinline)) atan_d_scalar(double x) {

    //bool xBig = (as_ulong(fabs(x)) > as_ulong(1.0));
    bool xBig = (fabs(x) > 1.0);

    double xReduced = x;

    if (xBig) {
        xReduced = 1.0 / x;
    }

    // We evaluate the polynomial using the Estrin scheme
    double x2 = xReduced * xReduced;
    double x4 = x2 * x2;
    double x8 = x4 * x4;
    double x16 = x8 * x8;

    // First layer of Estrin
    double L1 = FMA(x2, C3, C2);
    double L2 = FMA(x2, C5, C4);
    double L3 = FMA(x2, C7, C6);
    double L4 = FMA(x2, C9, C8);
    double L5 = FMA(x2, C11, C10);
    double L6 = FMA(x2, C13, C12);
    double L7 = FMA(x2, C15, C14);
    double L8 = FMA(x2, C17, C16);
    double L9 = FMA(x2, C19, C18);

    // We now want:
    // L1 + x4*L2 + x8*L3 + x12*L4 + x16*L5 + x20*L6 + x24*L7 + x28*L8 + x32*L9
    // + x36*C20 =
    //(L1 + x4*L2) + x8*(L3 + x4*L4) + x16*(L5 + x4*L6) + x24*(L7 + x4*L8) +
    //x32(*L9 + x4*C20)

    // Second layer of estrin
    double M1 = FMA(x4, L2, L1);
    double M2 = FMA(x4, L4, L3);
    double M3 = FMA(x4, L6, L5);
    double M4 = FMA(x4, L8, L7);
    double M5 = FMA(x4, C20, L9);

    // We now want:
    //  M1 + x8*M2 + x16*M3 + x24*M4 + x32*M5
    // (M1 + x8*M2) + x16*(M3 + x8*M4 + x16*M5)

    double N1 = FMA(x8, M2, M1);
    double N2 = FMA(x16, M5, M3 + x8 * M4);

    double poly = FMA(x16, N2, N1);

    if (xBig) {
        const double signedPi = copysign(PI_2, x);

        double result_d = FMA(-x2 * xReduced, poly, (signedPi - xReduced));

        return result_d;
    }

    double result_d = FMA(x2 * xReduced, poly, xReduced);

    result_d = copysign(result_d, x);

    return result_d;
}
