/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
#if defined COMPILING_EXP2
MATH_MANGLE(exp2)(double x)
#elif defined COMPILING_EXP10
MATH_MANGLE(exp10)(double x)
#else
MATH_MANGLE(exp)(double x)
#endif
{
#if defined(COMPILING_EXP2)
    double dn = BUILTIN_RINT_F64(x);
    double f = x - dn;
    double t = MATH_MAD(f, 0x1.62e42fefa39efp-1, f * 0x1.abc9e3b39803fp-56);
#elif defined(COMPILING_EXP10)
    double dn = BUILTIN_RINT_F64(x * 0x1.a934f0979a371p+1);
    double f = MATH_MAD(-dn, -0x1.9dc1da994fd21p-59, MATH_MAD(-dn, 0x1.34413509f79ffp-2, x));
    double t = MATH_MAD(f, 0x1.26bb1bbb55516p+1, f * -0x1.f48ad494ea3e9p-53);
#else
    double dn = BUILTIN_RINT_F64(x * 0x1.71547652b82fep+0);
    double t = MATH_MAD(-dn, 0x1.abc9e3b39803fp-56, MATH_MAD(-dn, 0x1.62e42fefa39efp-1, x));
#endif

    double p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                   0x1.ade156a5dcb37p-26, 0x1.28af3fca7ab0cp-22), 0x1.71dee623fde64p-19), 0x1.a01997c89e6b0p-16),
                   0x1.a01a014761f6ep-13), 0x1.6c16c1852b7b0p-10), 0x1.1111111122322p-7), 0x1.55555555502a1p-5),
                   0x1.5555555555511p-3), 0x1.000000000000bp-1), 1.0), 1.0);


    double z = BUILTIN_FLDEXP_F64(p, (int)dn);

    if (!FINITE_ONLY_OPT()) {
        z = x > 1024.0 ? PINF_F64 : z;
    }

    z = x < -1075.0 ? 0.0 : z;

    return z;
}

