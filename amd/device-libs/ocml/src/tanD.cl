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
    struct redret r = MATH_PRIVATE(trigred)(BUILTIN_ABS_F64(x));

    int2 t = AS_INT2(MATH_PRIVATE(tanred2)(r.hi, r.lo, r.i & 1));
    t.hi ^= AS_INT2(x).hi & (int)0x80000000;

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_INT2(QNANBITPATT_DP64) : t;
    }

    return AS_DOUBLE(t);
}

