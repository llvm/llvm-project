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
    double2 t = mul(x - dn, con(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56));
#elif defined(COMPILING_EXP10)
    double dn = BUILTIN_RINT_F64(x * 0x1.a934f0979a371p+1);
    double2 t = fsub(sub(mul(x, con(0x1.26bb1bbb55516p+1, -0x1.f48ad494ea3e9p-53)),
                         dn*0x1.62e42fefa3000p-1), dn*0x1.3de6af278ece6p-42);
#else
    double dn = BUILTIN_RINT_F64(x * 0x1.71547652b82fep+0);
    double2 t = fsub(MATH_MAD(dn, -0x1.62e42fefa3000p-1, x), dn*0x1.3de6af278ece6p-42);
#endif

    double th = t.hi;
    double p = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, 
               MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, 
               MATH_MAD(th, 
                   0x1.ade156a5dcb37p-26, 0x1.28af3fca7ab0cp-22), 0x1.71dee623fde64p-19), 0x1.a01997c89e6b0p-16),
                   0x1.a01a014761f6ep-13), 0x1.6c16c1852b7b0p-10), 0x1.1111111122322p-7), 0x1.55555555502a1p-5),
                   0x1.5555555555511p-3), 0x1.000000000000bp-1);

    double2 r = fadd(t, mul(sqr(t), p));
    double z = 1.0 + r.hi;

    z = BUILTIN_FLDEXP_F64(z, (int)dn);

    if (!FINITE_ONLY_OPT()) {
        z = x > 1024.0 ? AS_DOUBLE(PINFBITPATT_DP64) : z;
    }

    z = x < -1075.0 ? 0.0 : z;

    return z;
}

