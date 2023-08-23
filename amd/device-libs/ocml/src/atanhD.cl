/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double MATH_PRIVATE(lnep)(double2 a, int ea);

CONSTATTR double
MATH_MANGLE(atanh)(double x)
{
    double y = BUILTIN_ABS_F64(x);
    double2 a = fdiv(fadd(1.0, y), fsub(1.0, y));
    double z = 0.5 * MATH_PRIVATE(lnep)(a, 0);
    z = y < 0x1.0p-27 ? y : z;

    if (!FINITE_ONLY_OPT()) {
        z = y > 1.0 ? QNAN_F64 : z;
        z = y == 1.0 ? PINF_F64 : z;
    }

    return BUILTIN_COPYSIGN_F64(z, x);
}

