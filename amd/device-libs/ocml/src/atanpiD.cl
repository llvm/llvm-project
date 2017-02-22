/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double MATH_PRIVATE(atanred)(double);

CONSTATTR double
MATH_MANGLE(atanpi)(double x)
{
    const double pi = 0x1.921fb54442d18p+1;
    
    double v = BUILTIN_ABS_F64(x);
    bool g = v > 1.0;

    if (g) {
        v = MATH_RCP(v);
    }

    double a = MATH_PRIVATE(atanred)(v);
    a = MATH_DIV(a, pi);

    double y = 0.5 - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F64(a, x);
}

