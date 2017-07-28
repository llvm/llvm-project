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
MATH_PRIVATE(lnep)(double2 a)
{
    int b = BUILTIN_FREXP_MANT_F64(a.hi) < (2.0/3.0);
    int e = BUILTIN_FREXP_EXP_F64(a.hi) - b;
    double2 m = ldx(a, -e);
    double2 x = div(sub(m,1.0), add(m, 1.0));
    double2 s = sqr(x);
    double t = s.hi;
    double p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   0x1.dee674222de17p-4, 0x1.a6564968915a9p-4), 0x1.e25e43abe935ap-4), 0x1.110ef47e6c9c2p-3),
                   0x1.3b13bcfa74449p-3), 0x1.745d171bf3c30p-3), 0x1.c71c71c7792cep-3), 0x1.24924924920dap-2),
                   0x1.999999999999cp-2);

    // ln(2)*e + 2*x + x^3(c3 + x^2*p)
    double2 r = add(mul(con(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56), (double)e),
                    fadd(ldx(x,1),
                         mul(mul(s, x), 
                             fadd(con(0x1.5555555555555p-1,0x1.543b0d5df274dp-55),
                                  mul(s, p)))));

    return r.hi;
}

