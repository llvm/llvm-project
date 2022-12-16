/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double MATH_PRIVATE(lnep)(double2 a, int ea);

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double
MATH_MANGLE(log1p)(double x)
{
    double z = MATH_PRIVATE(lnep)(add(1.0, x), 0);

    if (!FINITE_ONLY_OPT()) {
        z = x == PINF_F64 ? x : z;
        z = x < -1.0 ? QNAN_F64 : z;
        z = x == -1.0 ? NINF_F64 : z;
    }

    return z;
}

