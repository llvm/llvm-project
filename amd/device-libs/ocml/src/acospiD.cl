/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double
MATH_MANGLE(acospi)(double x)
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
                       0x1.547a51d41fb0bp-7, -0x1.6a3fb0718a8f7p-8), 0x1.a7b91f7177ee8p-8), 0x1.035d3435b8ad8p-9),
                       0x1.ff0549b4e0449p-9), 0x1.21604ae288f96p-8), 0x1.6a2b36f9aec49p-8), 0x1.d2b076c914f04p-8),
                       0x1.3ce53861f8f1fp-7), 0x1.d1a4529a30a69p-7), 0x1.8723a1d61d2e9p-6), 0x1.b2995e7b7af0fp-5);

    const double piinv = 0x1.45f306dc9c883p-2;
    double z = 0.5 - MATH_MAD(x, u, piinv*x);
    if (transform) {
        double2 s = ldx(root2(r), 1);
        double zm = 1.0 - MATH_MAD(s.hi, u, piinv*s.hi);
        double2 zp = fadd(mul(piinv, s), mul(s, u));
        z = x < 0.0 ? zm : zp.hi;
        z = x == -1.0 ? 1.0 : z;
        z = x == 1.0 ? 0.0 : z;
    }

    return z;
}

