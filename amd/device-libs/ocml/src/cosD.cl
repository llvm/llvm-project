/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

double
MATH_MANGLE(cos)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);
    sc.s = -sc.s;

    long c = AS_LONG((r.i & 1) != 0 ? sc.s : sc.c);
    c ^= r.i > 1 ? SIGNBIT_DP64 : 0;

    double s = AS_DOUBLE(c);

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F64(ax) ? s : QNAN_F64;
    }

    return s;
}

