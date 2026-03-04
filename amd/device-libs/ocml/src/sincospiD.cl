/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

double
MATH_MANGLE(sincospi)(double x, __private double * cp)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F64(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    long flip = r.i > 1 ? SIGNBIT_DP64 : 0;
    bool odd = (r.i & 1) != 0;

    double s = odd ? sc.c : sc.s;

    s = AS_DOUBLE(AS_LONG(s) ^ flip ^ (AS_LONG(x) & SIGNBIT_DP64));
    sc.s = -sc.s;

    double c = odd ? sc.s : sc.c;
    c = AS_DOUBLE(AS_LONG(c) ^ flip);

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F64(x);
        s = finite ? s : QNAN_F64;
        c = finite ? c : QNAN_F64;
    }

    *cp = c;
    return s;
}

