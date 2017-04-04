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
MATH_MANGLE(acosh)(double x)
{
    bool b = x >= 0x1.0p+512;
    double s = b ? 0x1.0p-512 : 1.0;
    double sx = x * s;
    double2 a = add(sx, root2(sub(sqr(sx), s*s)));
    double z = MATH_PRIVATE(lnep)(a) + (b ? 0x1.62e42fefa39efp+8 : 0.0);

    z = x == 1.0 ? 0.0 : z;

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F64(x, CLASS_PINF) ? x : z;
        z = x < 1.0 ? AS_DOUBLE(QNANBITPATT_DP64) : z;
    }

    return z;
}

