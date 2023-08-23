/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

CONSTATTR double
MATH_PRIVATE(lnep)(double2 a, int ea)
{
    int b = BUILTIN_FREXP_MANT_F64(a.hi) < (2.0/3.0);
    int e = BUILTIN_FREXP_EXP_F64(a.hi) - b;
    double2 m = ldx(a, -e);
    double2 x = div(fadd(-1.0, m), fadd(1.0, m));
    double s = x.hi * x.hi;
    double p = MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s,
               MATH_MAD(s, MATH_MAD(s,
                   0x1.3ab76bf559e2bp-3, 0x1.385386b47b09ap-3), 0x1.7474dd7f4df2ep-3), 0x1.c71c016291751p-3),
                   0x1.249249b27acf1p-2), 0x1.99999998ef7b6p-2), 0x1.5555555555780p-1);
    double2 r = add(mul(con(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56), (double)(e + ea)),
                    fadd(ldx(x,1), s * x.hi * p));
    return r.hi;
}

