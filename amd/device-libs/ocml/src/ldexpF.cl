
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(ldexp)(float x, int n)
{
    if (AMD_OPT()) {
        return BUILTIN_FLDEXP_F32(x, n);
    } else if (DAZ_OPT()) {
        int ix = as_int(x);
        int e = (int)(ix >> EXPSHIFTBITS_SP32) & 0xff;
        n = BUILTIN_MIN_S32(BUILTIN_MAX_S32(n, -1024), 1024);
        int enew = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e + n, 0), 0xff);
        int m = ix & MANTBITS_SP32;
        int mnew = e == 0 | enew == 0 | enew == 0xff ? 0 : m;
        enew = e == 0 | e == 0xff ? e : enew;
        mnew = e == 0xff ? m : mnew;
        return as_float((ix & SIGNBIT_SP32) | (enew << EXPSHIFTBITS_SP32) | mnew);
    } else {
        uint ux = as_uint(x) & EXSIGNBIT_SP32;
        int e = (int)(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
        int es = -118 - (int)MATH_CLZI(ux);
        uint m = ux & MANTBITS_SP32;
        uint ms = ux << (-126 - es);
        m = e == -EXPBIAS_SP32 ? ms : m;
        e = e == -EXPBIAS_SP32 ? es : e;
        n = BUILTIN_MIN_S32(BUILTIN_MAX_S32(n, -1024), 1024);
        int en = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e + n, -160), 138);
        int enh = en >> 1;
        float ret = as_float((as_uint(x) ^ ux) | (EXPBIAS_SP32 << EXPSHIFTBITS_SP32) | m);
        ret *= as_float((EXPBIAS_SP32 + enh) << EXPSHIFTBITS_SP32);
        ret *= as_float((EXPBIAS_SP32 + (en - enh)) << EXPSHIFTBITS_SP32);
        ret = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_NZER|CLASS_PINF|CLASS_NINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
        return ret;
    }
}
