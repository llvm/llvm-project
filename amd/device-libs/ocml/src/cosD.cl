
#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(cos)(double x)
{
    x = BUILTIN_ABS_F64(x);

    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, x);

    double cc;
    double ss = -MATH_PRIVATE(sincosred2)(r, rr, &cc);

    int2 c = as_int2((regn & 1) != 0 ? ss : cc);
    c.hi ^= regn > 1 ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        return BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? as_double(QNANBITPATT_DP64) : as_double(c);
    } else {
	return as_double(c);
    }
}

