/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(tanpi)(double x)
{
    double r;
    int i = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F64(x), &r);

    int2 t = AS_INT2(MATH_PRIVATE(tanpired)(r, i & 1));
    t.hi ^= (((i == 1) | (i == 2)) & (r == 0.0)) ? 0x80000000 : 0;
    t.hi ^= AS_INT2(x).hi & (int)0x80000000;

    if (!FINITE_ONLY_OPT()) {
        t =  BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_INT2(QNANBITPATT_DP64) : t;
    }

    return AS_DOUBLE(t);
}

