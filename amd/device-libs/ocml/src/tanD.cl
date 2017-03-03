/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(tan)(double x)
{
    double r, rr;
    int i = MATH_PRIVATE(trigred)(&r, &rr, BUILTIN_ABS_F64(x));

    int2 t = AS_INT2(MATH_PRIVATE(tanred2)(r, rr, i & 1));
    t.hi ^= AS_INT2(x).hi & (int)0x80000000;

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_INT2(QNANBITPATT_DP64) : t;
    }

    return AS_DOUBLE(t);
}

