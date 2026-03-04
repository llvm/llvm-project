/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double MATH_PRIVATE(lnep)(double2 a, int ea);

CONSTATTR double2
MATH_MANGLE(catanh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double y = BUILTIN_ABS_F64(z.y);
    double rr, ri;

    if (x < 0x1.0p+54 && y < 0x1.0p+54) {
        double2 omx = sub(1.0, x);
        double2 opx = add(1.0, x);
        double2 y2 = sqr(y);
        double2 b = sub(mul(omx, opx), y2);
        ri = 0.5 * MATH_MANGLE(atan2)(2.0 * y, b.hi);

        double2 a;
        double2 d = add(sqr(opx), y2);
        if (x < 0x1.0p-3 * d.hi) {
            a = fsub(1.0, div(4.0*x, d));
        } else {
            a = div(add(sqr(omx), y2), d);
        }
        rr = -0.25 * MATH_PRIVATE(lnep)(a, 0);
    } else {
        int e = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(x, y));
        x = BUILTIN_FLDEXP_F64(x, -e);
        y = BUILTIN_FLDEXP_F64(y, -e);
        rr = BUILTIN_FLDEXP_F64(MATH_DIV(x, MATH_MAD(x, x, y*y)), -e);
        ri = 0x1.921fb54442d18p+0;
    }

    if (!FINITE_ONLY_OPT()) {
        rr = ((x == 1.0) & (y == 0.0)) ? PINF_F64  : rr;
        rr = x == 0.0 ? 0.0 : rr;
        rr = BUILTIN_ISINF_F64(x) ? 0.0 : rr;
        rr = (BUILTIN_ISNAN_F64(x) & BUILTIN_ISINF_F64(y)) ? 0.0 : rr;
        ri = (BUILTIN_ISNAN_F64(x) & BUILTIN_ISFINITE_F64(y)) ? QNAN_F64 : ri;
        ri = BUILTIN_ISNAN_F64(y) ? y : ri;
    }

    rr = BUILTIN_COPYSIGN_F64(rr, z.x);
    ri = BUILTIN_COPYSIGN_F64(ri, z.y);

    return (double2)(rr, ri);
}

