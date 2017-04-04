/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double MATH_PRIVATE(lnep)(double2 x);

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(log1p)(double x)
{
    double z = MATH_PRIVATE(lnep)(add(1.0, x));

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F64(x, CLASS_PINF) ? x : z;
        z = x < -1.0 ? AS_DOUBLE(QNANBITPATT_DP64) : z;
        z = x == -1.0 ? AS_DOUBLE(NINFBITPATT_DP64) : z;
    }

    return z;
}

