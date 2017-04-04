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
MATH_MANGLE(asinh)(double x)
{
    double y = BUILTIN_ABS_F64(x);
    bool b = y >= 0x1.0p+512;
    double s = b ? 0x1.0p-512 : 1.0;
    double sy = y * s;
    double2 a = add(sy, root2(add(sqr(sy), s*s)));
    double z = MATH_PRIVATE(lnep)(a) + (b ? 0x1.62e42fefa39efp+8 : 0.0);

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F64(y, CLASS_PINF) ? y : z;
    }

    return BUILTIN_COPYSIGN_F64(z, x);
}

