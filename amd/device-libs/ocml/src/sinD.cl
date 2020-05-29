/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR double
MATH_MANGLE(sin)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);

    int2 s = AS_INT2((r.i & 1) == 0 ? sc.s : sc.c);
    s.hi ^= (r.i > 1 ? 0x80000000 : 0) ^ (AS_INT2(x).hi & 0x80000000);

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F64(ax) ? s : AS_INT2(QNANBITPATT_DP64);
    }

    return AS_DOUBLE(s);
}

