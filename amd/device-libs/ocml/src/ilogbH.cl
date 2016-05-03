
#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(ilogb)(half x)
{
    int r;

    if (AMD_OPT()) {
        r = BUILTIN_FREXP_EXP_F16(x) - 1;
    } else {
        int ix = (int)as_ushort(x);
        r = ((ix >> 10) & 0x1f) - EXPBIAS_HP16;
        int rs = 7 - (int)MATH_CLZI(ix & MANTBITS_HP16);
        r = BUILTIN_CLASS_F16(x, CLASS_PSUB|CLASS_NSUB) ? rs : r;
    }

    if (!FINITE_ONLY_OPT()) {
        r = (x == 0.0h) | BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN) ? (int)0x80000000 : r;
        r = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) ? 0x7fffffff : r;
    } else {
	r = x == 0.0h ? 0x80000000 : r;
    }

    return r;
}

