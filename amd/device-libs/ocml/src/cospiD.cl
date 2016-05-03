
#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(cospi)(double x)
{
    x = BUILTIN_ABS_F64(x);
    double r = BUILTIN_FRACTION_F64(x);
    double txh = BUILTIN_TRUNC_F64(x) * 0.5;
    double sgn = txh != BUILTIN_TRUNC_F64(txh) ? -1.0 : 1.0;
    double ret;

    // 2^53 <= |x| < Inf, the result is always even integer
    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF) ? as_double(QNANBITPATT_DP64) : 1.0;
    } else {
	ret = 1.0;
    }

    // 2^52 <= |x| < 2^53, the result is always integer
    ret = x < 0x1.0p+53 ? sgn : ret;

    // 0x1.0p-7 <= |x| < 2^52, result depends on which 0.25 interval

    // r < 1.0
    double a = 1.0 - r;
    int e = 1;
    double s = -sgn;

    // r <= 0.75
    bool c = r <= 0.75;
    double t = r - 0.5;
    a = c ? t : a;
    e = c ? 0 : e;

    // r < 0.5
    c = r < 0.5;
    t = 0.5 - r;
    a = c ? t : a;
    s = c ? sgn : s;

    // r <= 0.25
    c = r <= 0.25;
    a = c ? r : a;
    e = c ? 1 : e;

    const double pi = 0x1.921fb54442d18p+1;
    double ca;
    double sa = MATH_PRIVATE(sincosred)(a * pi, &ca);

    double tret = BUILTIN_COPYSIGN_F64(e ? ca : sa, s);
    ret = x < 0x1.0p+52 ? tret : ret;

    return ret;
}

