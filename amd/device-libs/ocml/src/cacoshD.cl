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
MATH_MANGLE(cacosh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double y = BUILTIN_ABS_F64(z.y);

    double2 l2, t;
    int e = 0;
    bool b = true;

    if (x < 0x1.0p+54 && y < 0x1.0p+54) {
        if (x >= 1.0 || y >= 0x1.0p-53 || y > (1.0 - x)*0x1.0p-26) {
            double4 z2p1 = (double4)(add(mul(add(y,x), sub(y,x)), 1.0), mul(y,x)*2.0);
            double4 rz2m1 = MATH_PRIVATE(epcsqrtep)(z2p1);
            rz2m1 = (double4)(csgn(rz2m1.hi, (double2)z.x), csgn(rz2m1.lo, (double2)z.y));
            double4 s = (double4)(add(rz2m1.lo, z.x), add(rz2m1.hi, z.y));
            l2 = add(sqr(s.lo), sqr(s.hi));
            t = (double2)(s.s1, z.y == 0.0 ? z.y : s.s3);
        } else {
            b = false;
            double r = MATH_FAST_SQRT(BUILTIN_FMA_F64(-x, x, 1.0));
            l2 = con(MATH_DIV(y, r), 0.0);
            t = (double2)(z.x, BUILTIN_COPYSIGN_F64(r, z.y));
        }
    } else {
        e = BUILTIN_FREXP_EXP_F64(BUILTIN_MAX_F64(x,y));
        x = BUILTIN_FLDEXP_F64(x, -e);
        y = BUILTIN_FLDEXP_F64(y, -e);
        l2 = add(sqr(x), sqr(y));
        e = 2*e + 2;
        t = z;
    }

    double rr;
    if (b) {
        rr = 0.5 * MATH_PRIVATE(lnep)(l2, e);
    } else {
        rr = l2.hi;
    }

    double ri = MATH_MANGLE(atan2)(t.y, t.x);

    if (!FINITE_ONLY_OPT()) {
        rr = (BUILTIN_ISINF_F64(z.x) | BUILTIN_ISINF_F64(z.y)) ? PINF_F64 : rr;
    }

    return (double2)(rr, ri);
}

