
#include "mathF.h"

INLINEATTR float
MATH_MANGLE(frexp)(float x, __private int *ep)
{
    if (AMD_OPT()) {
        int e = BUILTIN_FREXP_EXP_F32(x);
        float r = BUILTIN_FREXP_MANT_F32(x);
        bool c = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF|CLASS_SNAN|CLASS_QNAN);
        *ep = c ? 0 : e;
        return c ? x : r;
    } else if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
        int i = AS_INT(x);
        int ai = i & EXSIGNBIT_SP32;
        int e = (ai >> 23) - 126;
        bool t = ai == 0 | e == 129;
        i = (i & SIGNBIT_SP32) | HALFEXPBITS_SP32 | (ai & MANTBITS_SP32);
        *ep = t ? 0 : e;
        return t ? x : AS_FLOAT(i);
    } else {
        int i = AS_INT(x);
        int ai = i & EXSIGNBIT_SP32;
        bool d = ai > 0 & ai < IMPBIT_SP32;
        float s = AS_FLOAT(ONEEXPBITS_SP32 | ai) - 1.0f;
        ai = d ? AS_INT(s) : ai;
        int e = (ai >> EXPSHIFTBITS_SP32) - (d ? 252 : 126);
        bool t = ai == 0 | e == 129;
        i = (i & SIGNBIT_SP32) | HALFEXPBITS_SP32 | (ai & MANTBITS_SP32);
        *ep = t ? 0 : e;
        return t ? x : AS_FLOAT(i);
    }
}

