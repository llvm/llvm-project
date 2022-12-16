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
MATH_PRIVATE(expep)(double2 x)
{
#if defined EXTRA_ACCURACY
    double dn = BUILTIN_RINT_F64(x.hi * 0x1.71547652b82fep+0);
    double2 t = fsub(fsub(sub(x, dn*0x1.62e42fefa3000p-1), dn*0x1.3de6af278e000p-42), dn*0x1.9cc01f97b57a0p-83);

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

    z = x.hi > 710.0 ? PINF_F64 : z;
    z = x.hi < -745.0 ? 0.0 : z;
#else
    double z = MATH_MANGLE(exp)(x.hi);
    double zz = MATH_MAD(z, x.lo, z);
    z = BUILTIN_ISINF_F64(z)? z : zz;
#endif

    return z;
}

