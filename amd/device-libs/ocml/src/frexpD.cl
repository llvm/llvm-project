
#include "mathD.h"

INLINEATTR double
MATH_MANGLE(frexp)(double x, __private int *ep)
{
    if (AMD_OPT()) {
        int e = BUILTIN_FREXP_EXP_F64(x);
        double r = BUILTIN_FREXP_MANT_F64(x);
        bool c = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF|CLASS_SNAN|CLASS_QNAN);
        *ep = c ? 0 : e;
        return c ? x : r;
    } else {
        long i = as_long(x);
        long ai = i & EXSIGNBIT_DP64;
        bool d = ai > 0 & ai < IMPBIT_DP64;
        double s = as_double(ONEEXPBITS_DP64 | ai) - 1.0;
        ai = d ? as_long(s) : ai;
        int e = (int)(as_int2(ai).hi >> 20) - (d ? 2044 : 1022);
        bool t = ai == 0 | e == 1025;
        i = (i & SIGNBIT_DP64) | HALFEXPBITS_DP64 | (ai & MANTBITS_DP64);
        *ep = t ? 0 : e;
        return t ? x : as_double(i);
    }
}

