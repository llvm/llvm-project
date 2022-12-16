/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double4 MATH_PRIVATE(epcsqrtep)(double4 z);
extern CONSTATTR double MATH_PRIVATE(lnep)(double2 a, int ea);

CONSTATTR double2
MATH_MANGLE(casinh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double y = BUILTIN_ABS_F64(z.y);

    double2 l2, t;
    int e = 0;
    bool b = true;

    if (x < 0x1.0p+54 && y < 0x1.0p+54) {
        if (y >= 1.0 || x >= 0x1.0p-53 || x > (1.0 - y)*0x1.0p-26f) {
            double4 z2p1 = (double4)(add(mul(add(x,y), sub(x,y)), 1.0), mul(y,x)*2.0);
            double4 rz2p1 = MATH_PRIVATE(epcsqrtep)(z2p1);
            double4 s = (double4)(add(rz2p1.lo, x), add(rz2p1.hi, y));
            l2 = add(sqr(s.lo), sqr(s.hi));
            t = (double2)(s.s1, s.s3);
        } else {
            b = false;
            double r = MATH_SQRT(BUILTIN_FMA_F64(-y, y, 1.0));
            l2 = con(MATH_DIV(x, r), 0.0);
            t = (double2)(r, y);
        }
    } else {
        t = (double2)(x, y);
        e = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(x, y));
        x = BUILTIN_FLDEXP_F64(x, -e);
        y = BUILTIN_FLDEXP_F64(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
    }

    double rr;
    if (b) {
        rr = 0.5 * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    rr = BUILTIN_COPYSIGN_F64(rr, z.x);
    double ri = BUILTIN_COPYSIGN_F64(MATH_MANGLE(atan2)(t.y, t.x), z.y);

    if (!FINITE_ONLY_OPT()) {
        double i = BUILTIN_COPYSIGN_F64(PINF_F64, z.x);
        rr = (BUILTIN_ISINF_F64(z.x) | BUILTIN_ISINF_F64(z.y)) ? i : rr;
    }

    return (double2)(rr, ri);
}

