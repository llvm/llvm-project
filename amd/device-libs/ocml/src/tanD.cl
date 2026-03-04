/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR double
MATH_MANGLE(tan)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);

    double t = MATH_PRIVATE(tanred2)(r.hi, r.lo, r.i & 1);
    t = AS_DOUBLE(AS_LONG(t) ^ (AS_LONG(x) & SIGNBIT_DP64));

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISFINITE_F64(ax) ? t : QNAN_F64;
    }

    return AS_DOUBLE(t);
}

