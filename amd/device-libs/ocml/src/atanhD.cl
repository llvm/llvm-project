/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double MATH_PRIVATE(lnep)(double2 x);

CONSTATTR INLINEATTR double
MATH_MANGLE(atanh)(double x)
{
    double y = BUILTIN_ABS_F64(x);
    double2 a = fdiv(fadd(1.0, y), fsub(1.0, y));
    double z = 0.5 * MATH_PRIVATE(lnep)(a);
    z = y < 0x1.0p-27 ? y : z;

    if (!FINITE_ONLY_OPT()) {
        z = y > 1.0 ? AS_DOUBLE(QNANBITPATT_DP64) : z;
        z = y == 1.0 ? AS_DOUBLE(PINFBITPATT_DP64) : z;
    }

    return BUILTIN_COPYSIGN_F64(z, x);
}

