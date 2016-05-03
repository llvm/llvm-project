
#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(tan)(double x)
{
    double y = BUILTIN_ABS_F64(x);

    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, y);

    double tt = MATH_PRIVATE(tanred2)(r, rr, regn & 1);
    int2 t = as_int2(tt);
    t.hi ^= x < 0.0 ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        return BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? as_double(QNANBITPATT_DP64) : as_double(t);
    } else {
	return as_double(t);
    }
}

