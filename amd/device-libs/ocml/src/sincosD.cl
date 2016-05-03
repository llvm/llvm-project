
#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(sincos)(double x, __private double * cp)
{
    double y = BUILTIN_ABS_F64(x);

    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, y);

    double cc;
    double ss = MATH_PRIVATE(sincosred2)(r, rr, &cc);

    int flip = regn > 1 ? (int)0x80000000 : 0;

    int2 s = as_int2((regn & 1) != 0 ? cc : ss);
    s.hi ^= flip ^ (x < 0.0 ? (int)0x80000000 : 0);
    ss = -ss;
    int2 c = as_int2(regn & 1 ? ss : cc);
    c.hi ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool xgeinf = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF);
        s = xgeinf ? as_int2(QNANBITPATT_DP64) : s;
        c = xgeinf ? as_int2(QNANBITPATT_DP64) : c;
    }

    *cp = as_double(c);
    return as_double(s);
}

