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
    double y = BUILTIN_ABS_F64(x);

    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, y);

    double tt = MATH_PRIVATE(tanred2)(r, rr, regn & 1);
    int2 t = AS_INT2(tt);
    t.hi ^= x < 0.0 ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        return BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_DOUBLE(QNANBITPATT_DP64) : AS_DOUBLE(t);
    } else {
	return AS_DOUBLE(t);
    }
}

