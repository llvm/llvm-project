
#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(sin)(double x)
{
    double y = BUILTIN_ABS_F64(x);

    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, y);

    double cc;
    double ss = MATH_PRIVATE(sincosred2)(r, rr, &cc);

    int2 s = AS_INT2((regn & 1) != 0 ? cc : ss);
    s.hi ^= (regn > 1) ^ (x < 0.0) ? (int)0x80000000 : 0;

    if (!FINITE_ONLY_OPT()) {
        return BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? AS_DOUBLE(QNANBITPATT_DP64) : AS_DOUBLE(s);
    } else {
	return AS_DOUBLE(s);
    }
}

