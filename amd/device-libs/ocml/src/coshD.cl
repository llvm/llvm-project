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
MATH_MANGLE(cosh)(double x)
{
    x = BUILTIN_ABS_F64(x);
    double2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e42fefa39efp-1,0x1.abc9e3b39803fp-56)));
    double2 c = fadd(e, ldx(rcp(e), -2));
    double z = c.hi;
    
    if (!FINITE_ONLY_OPT()) {
        z = x >= 0x1.633ce8fb9f87ep+9 ? PINF_F64 : z;
    }

    return z;
}
  
