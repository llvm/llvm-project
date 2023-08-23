/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

double
MATH_MANGLE(sincos)(double x, __private double * cp)
{
    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);

    int flip = r.i > 1 ? (int)0x80000000 : 0;
    bool odd = (r.i & 1) != 0;

    int2 s = AS_INT2(odd ? sc.c : sc.s);
    s.hi ^= flip ^ (AS_INT2(x).hi &(int)0x80000000);
    sc.s = -sc.s;
    int2 c = AS_INT2(odd ? sc.s : sc.c);
    c.hi ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F64(x);
        s = finite ? s : AS_INT2(QNANBITPATT_DP64);
        c = finite ? c : AS_INT2(QNANBITPATT_DP64);
    }

    *cp = AS_DOUBLE(c);
    return AS_DOUBLE(s);
}

