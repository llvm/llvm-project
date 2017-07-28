/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

double
MATH_MANGLE(cospi)(double x)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F64(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);
    sc.s = -sc.s;

    int2 c = AS_INT2((r.i & 1) == 0 ? sc.c : sc.s);
    c.hi ^= r.i > 1 ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_INT2(QNANBITPATT_DP64) : c;
    }

    return AS_DOUBLE(c);
}

