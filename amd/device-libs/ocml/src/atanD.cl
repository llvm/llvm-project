/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double MATH_PRIVATE(atanred)(double);

CONSTATTR double
MATH_MANGLE(atan)(double x)
{
    double v = BUILTIN_ABS_F64(x);
    bool g = v > 1.0;

    if (g) {
        v = MATH_RCP(v);
    }

    double a = MATH_PRIVATE(atanred)(v);

    double y = BUILTIN_FMA_F64(0x1.dd9ad336a0500p-1, 0x1.af154eeb562d6p+0, -a);
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F64(a, x);
}

