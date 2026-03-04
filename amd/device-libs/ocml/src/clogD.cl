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
MATH_MANGLE(clog)(double2 z)
{
    double x = z.s0;
    double y = z.s1;
    double a = BUILTIN_ABS_F64(x);
    double b = BUILTIN_ABS_F64(y);
    double t = BUILTIN_MAX_F64(a, b);
    int e = BUILTIN_FREXP_EXP_F64(t) ;
    a = BUILTIN_FLDEXP_F64(a, -e);
    b = BUILTIN_FLDEXP_F64(b, -e);
    double rr = 0.5 * MATH_PRIVATE(lnep)(add(sqr(a), sqr(b)), 2*e);
    double ri = MATH_MANGLE(atan2)(y, x);
    

    if (!FINITE_ONLY_OPT()) {
        rr = ((x == 0.0) & (y == 0.0)) ? NINF_F64 : rr;
        rr = (BUILTIN_ISINF_F64(x) | BUILTIN_ISINF_F64(y)) ? PINF_F64 : rr;
    }

    return (double2)(rr, ri);
}

