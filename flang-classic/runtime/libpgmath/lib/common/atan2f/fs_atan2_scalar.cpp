
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <common.h>
#include <math.h>

#define _JOIN2(a,b) a##b
#define JOIN2(a,b) _JOIN2(a,b)

#define atan2_scalar JOIN2(__fs_atan2_1_,_CPU)
#define FMAF __builtin_fmaf

extern "C" float atan2_scalar(float,float);


static unsigned int __attribute__ ((always_inline)) as_uint(float x) 
{ 
    return *(unsigned int *)&x;
}

static float __attribute__ ((always_inline)) as_float(unsigned long long int x)
{ 
    return *(float *)&x;
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

float atan2_scalar(float const y, float const x) {

    // Special return values when both inputs are infinity, or 0's
    // (or any absolute equal numbers, the results are the same):
    if (__builtin_expect(fabs(y) == fabs(x), 1)) {

        // Using (as_uint(x) > 0x7FFFFFFF) rather than (x < 0.0) here
        // seems to give a performance boost when looping over this function
        // many times
        float ans = FMAF((as_uint(x) > 0x7FFFFFFF), PI_OVER_2, PI_OVER_4);

        // Special return values for (y, x) = (+-0.0, +-0.0)
        if (x == 0.0f) {
            ans = (as_uint(x) == 0x0) ? 0.0f : PI;
        }

        return copysign(ans, y);
    }

    float xReduced;

    // xReduced = (fabs(y) > fabs(x) ? x : y) / (fabs(y) > fabs(x) ? y : x);
    // Seems to be the fastest way of getting x/y or y/x  
    // Comparing these as unsinged int's is a bit faster:
    if (as_uint(fabs(y)) > as_uint(fabs(x))) {
       xReduced = x / y;
    } else {
        xReduced = y / x;
    }

    // The same Estrin scheme as is used in atan(x):
    float x2 = xReduced * xReduced;
    float x4 = x2 * x2;
    float x8 = x4 * x4;

    // First layer of Estrin:
    float L1 = FMAF(x2, C2, C1);
    float L2 = FMAF(x2, C4, C3);
    float L3 = FMAF(x2, C6, C5);
    float L4 = FMAF(x2, C8, C7);

    // Second layer of estrin
    float M1 = FMAF(x4, L2, L1);
    float M2 = FMAF(x4, L4, L3);

    float poly = FMAF(x8, M2, M1);

    float result_f = poly;

    float pi_factor = 0.0f;

    // Comparing these as unsinged int's is a bit faster:
    if (as_uint(fabs(y)) > as_uint(fabs(x))) {
        // pi/2 with the sign of xReduced:
        // Manually doing the copysign here seems to be faster:
        // const float signedPi_2 = copysignf(PI_OVER_2, xReduced);
        const float signedPi_2 =
            as_float(as_uint(PI_OVER_2) | (as_uint(xReduced) & 0x80000000));

        xReduced = -xReduced;
        pi_factor = signedPi_2;
    }

    result_f = FMAF(x2 * xReduced, poly, xReduced);

    // pi with the sign of y:
    // const float signedPi = as_float(as_uint(PI) | (as_uint(y) & 0x80000000));
    const float signedPi = copysignf(PI, y);

    // We need to check for -0.0 here as well as x < 0.0, so we cast to uint:
    // Again, faster:
    pi_factor = FMAF(as_uint(x) >= 0x80000000, signedPi, pi_factor);
    // if (as_uint(x) > 0x7FFFFFFF) {
    //    pi_factor += signedPi;
    //}

    result_f += pi_factor;

    // Unfortunately we need to do a copysign here because of the potential of y
    // to be -0.0, and we return an incorrectly signed 0.0
    result_f = copysignf(result_f, y);

    return result_f;
}

