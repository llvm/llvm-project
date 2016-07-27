/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(cos)(double x)
{
    x = BUILTIN_ABS_F64(x);

    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, x);

    double cc;
    double ss = -MATH_PRIVATE(sincosred2)(r, rr, &cc);

    int2 c = AS_INT2((regn & 1) != 0 ? ss : cc);
    c.hi ^= regn > 1 ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        return BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_DOUBLE(QNANBITPATT_DP64) : AS_DOUBLE(c);
    } else {
	return AS_DOUBLE(c);
    }
}

