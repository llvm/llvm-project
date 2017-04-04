/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 x);

CONSTATTR double
MATH_MANGLE(expm1)(double x)
{
    double2 e = sub(MATH_PRIVATE(epexpep)(con(x, 0.0)), 1.0);
    double z = e.hi;
    
    if (!FINITE_ONLY_OPT()) {
        z = x > 0x1.62e42fefa39efp+9 ? AS_DOUBLE(PINFBITPATT_DP64) : z;
    }

    z = x < -37.0 ? -1.0 : z;

    return z;
}

