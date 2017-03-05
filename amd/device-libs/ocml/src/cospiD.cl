/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

INLINEATTR double
MATH_MANGLE(cospi)(double x)
{
    double t;
    int i = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F64(x), &t);

    double cc;
    double ss = -MATH_PRIVATE(sincospired)(t, &cc);

    int2 c = AS_INT2((i & 1) == 0 ? cc : ss);
    c.hi ^= i > 1 ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_INT2(QNANBITPATT_DP64) : c;
    }

    return AS_DOUBLE(c);
}

