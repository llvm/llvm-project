
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(ldexp)(double x, int n)
{
    if (AMD_OPT()) {
        return BUILTIN_FLDEXP_F64(x, n);
    } else {
        int e = ((AS_INT2(x).hi >> 20) & 0x7ff) - EXPBIAS_DP64;
        double xs = x * 0x1.0p+53;
        int es = ((AS_INT2(xs).hi >> 20) & 0x7ff) - EXPBIAS_DP64 - 53;
        double t = e == -EXPBIAS_DP64 ? xs : x;
        e = e == -EXPBIAS_DP64 ? es : e;
        double ret = AS_DOUBLE(((long)EXPBIAS_DP64 << EXPSHIFTBITS_DP64) | (AS_LONG(t) & ~EXPBITS_DP64));
        n = BUILTIN_MIN_S32(BUILTIN_MAX_S32(n, -4096), 4096);
        int en = BUILTIN_MIN_S32(BUILTIN_MAX_S32(e + n, -1080), 1031);
        int enh = en >> 1;
        ret *= AS_DOUBLE((long)(EXPBIAS_DP64 + enh) << EXPSHIFTBITS_DP64);
        ret *= AS_DOUBLE((long)(EXPBIAS_DP64 + (en - enh)) << EXPSHIFTBITS_DP64);
        ret = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_NZER|CLASS_PINF|CLASS_NINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
        return ret;
    }
}
