/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(acos)(double x)
{
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    double y = BUILTIN_ABS_F64(x);
    bool transform = y >= 0.5;

    double rt = MATH_MAD(y, -0.5, 0.5);
    double y2 = y * y;
    double r = transform ? rt : y2;

    double u = r * MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, 
                   MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, 
                   MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, 
                       0x1.059859fea6a70p-5, -0x1.0a5a378a05eafp-6), 0x1.4052137024d6ap-6), 0x1.ab3a098a70509p-8),
                       0x1.8ed60a300c8d2p-7), 0x1.c6fa84b77012bp-7), 0x1.1c6c111dccb70p-6), 0x1.6e89f0a0adacfp-6),
                       0x1.f1c72c668963fp-6), 0x1.6db6db41ce4bdp-5), 0x1.333333336fd5bp-4), 0x1.5555555555380p-3);

    double z;
    if (transform) {
        double s = MATH_FAST_SQRT(r);
        if (x < 0.0) {
            // z =  MATH_MAD(-2.0, (s + MATH_MAD(s, u, -piby2_tail)), pi);
            z = MATH_MAD(0x1.dd9ad336a0500p+0, 0x1.af154eeb562d6p+0, -2.0*MATH_MAD(s, u, s));
            z = x == -1.0 ? 0x1.921fb54442d18p+1 : z;
        } else {
            // Compute higer precision square root
            double sh = AS_DOUBLE(AS_ULONG(s) & 0xffffffff00000000UL);
            double st = MATH_FAST_DIV(MATH_MAD(-sh, sh, r), s + sh);
            z = 2.0 * (sh + MATH_MAD(s, u, st));
            z = x == 1.0 ? 0.0 : z;
        }
    } else {
        // z = piby2_head - (x - MATH_MAD(-x, u, piby2_tail));
        z = MATH_MAD(0x1.dd9ad336a0500p-1, 0x1.af154eeb562d6p+0, -MATH_MAD(x, u, x));
    }

    return z;
}

