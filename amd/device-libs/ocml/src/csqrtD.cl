/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double2
MATH_MANGLE(csqrt)(double2 z)
{
    double a = BUILTIN_ABS_F64(z.x);
    double b = BUILTIN_ABS_F64(z.y);
    double t = BUILTIN_MAX_F64(a, b);

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISUNORDERED_F64(a, b) ? QNAN_F64 : t;
    }

    int e = BUILTIN_FREXP_EXP_F64(t);
    double as = BUILTIN_FLDEXP_F64(a, -e);
    double bs = BUILTIN_FLDEXP_F64(b, -e);
    bool o = BUILTIN_CLASS_F64(t, CLASS_NZER|CLASS_PZER|CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN);
    double p = MATH_FAST_SQRT(MATH_MAD(as, as, bs*bs));
    p = o ? t : p;
    int k = (e & 1) ^ 1; 
    p = BUILTIN_FLDEXP_F64(p + as, k);
    p = BUILTIN_FLDEXP_F64(MATH_FAST_SQRT(p), (e >> 1) - k);
    p = o ? t : p;
    double q = BUILTIN_FLDEXP_F64(MATH_DIV(b, p), -1);
    q = t == 0.0 ? t : q;
    bool l = z.x < 0.0;
    double rr = l ? q : p;
    double ri = l ? p : q;

    if (!FINITE_ONLY_OPT()) {
        bool i = BUILTIN_ISINF_F64(b);
        rr = i ? b : rr;
        ri = i ? b : ri;
        ri = z.x == NINF_F64 ? a : ri;
        rr = z.x == PINF_F64 ? a : rr;
    }

    return (double2)(rr, BUILTIN_COPYSIGN_F64(ri, z.y));
}

