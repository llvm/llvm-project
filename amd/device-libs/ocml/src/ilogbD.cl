
#include "mathD.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(ilogb)(double x)
{
    int r;

    if (AMD_OPT()) {
        r = BUILTIN_FREXP_EXP_F64(x) - 1;
    } else {
        r = ((AS_INT2(x).hi >> 20) & 0x7ff) - EXPBIAS_DP64;
        int rs = -1011 - (int)MATH_CLZL(AS_LONG(x) & MANTBITS_DP64);
        r = BUILTIN_CLASS_F64(x, CLASS_PSUB|CLASS_NSUB) ? rs : r;
    }

    if (!FINITE_ONLY_OPT()) {
        r = (x == 0.0) | BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? (int)0x80000000 : r;
        r = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF) ? 0x7fffffff : r;
    } else {
	r = x == 0.0 ? 0x80000000 : r;
    }

    return r;
}

