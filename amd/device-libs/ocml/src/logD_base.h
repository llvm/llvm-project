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
#if defined(COMPILING_LOG2)
MATH_MANGLE(log2)(double a)
#elif defined(COMPILING_LOG10)
MATH_MANGLE(log10)(double a)
#else
MATH_MANGLE(log)(double a)
#endif
{
    double m = BUILTIN_FREXP_MANT_F64(a);
    int b = m < (2.0/3.0);
    m = BUILTIN_FLDEXP_F64(m, b);
    int e = BUILTIN_FREXP_EXP_F64(a) - b;

    double2 x = div(m - 1.0, add(m, 1.0));
    double2 s = sqr(x);
    double t = s.hi;
    double p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
               MATH_MAD(t, MATH_MAD(t,
                   0x1.3ab76bf559e2bp-3, 0x1.385386b47b09ap-3), 0x1.7474dd7f4df2ep-3), 0x1.c71c016291751p-3),
                   0x1.249249b27acf1p-2), 0x1.99999998ef7b6p-2), 0x1.5555555555780p-1);
    double2 r = add(ldx(x,1), mul(mul(x,s), p));

#if defined COMPILING_LOG2
    r = add((double)e, mul(con(0x1.71547652b82fep+0,0x1.777d0ffda0d24p-56), r));
#elif defined COMPILING_LOG10
    r = add(mul(con(0x1.34413509f79ffp-2, -0x1.9dc1da994fd21p-59), (double)e),
            mul(con(0x1.bcb7b1526e50ep-2, 0x1.95355baaafad3p-57), r));
#else
    r = add(mul(con(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56), (double)e), r);
#endif

    double ret = r.hi;

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISINF_F64(a) ? a : ret;
        ret = a < 0.0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = a == 0.0 ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
    }

    return ret;
}

