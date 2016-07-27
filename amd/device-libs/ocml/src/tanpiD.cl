/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(tanpi)(double x)
{
    double z = BUILTIN_COPYSIGN_F64(0.0, x);
    x = BUILTIN_ABS_F64(x);
    double r = BUILTIN_FRACTION_F64(x);
    double txh = BUILTIN_TRUNC_F64(x) * 0.5;
    double sgn = (txh != BUILTIN_TRUNC_F64(txh)) ^ BUILTIN_CLASS_F64(z, CLASS_NZER) ? -0.0 : 0.0;
    double ret;

    // 2^53 <= |x| < Inf, the result is always even integer
    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF) ? AS_DOUBLE(QNANBITPATT_DP64) : z;
    } else {
	ret = z;
    }

    // 2^52 <= |x| < 2^53, the result is always integer
    ret = x < 0x1.0p+53 ? sgn : ret;

    // 0x1.0p-14 <= |x| < 2^53, result depends on which 0.25 interval

    // r < 1.0
    double a = 1.0 - r;
    int e = 0;
    double s = -z;

    // r <= 0.75
    bool c = r <= 0.75;
    double t = r - 0.5;
    a = c ? t : a;
    e = c ? 1 : e;
    s = c ? z : s;

    // r < 0.5
    c = r < 0.5;
    t = 0.5 - r;
    a = c ? t : a;
    s = c ? -z : s;

    // r <= 0.25
    c = r <= 0.25;
    a = c ? r : a;
    e = c ? 0 : e;
    s = c ? z : s;

    const double pi = 0x1.921fb54442d18p+1;
    double tret = MATH_PRIVATE(tanred2)(a * pi, 0.0, e) * BUILTIN_COPYSIGN_F64(1.0, s);
    double tinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), sgn);
    tret = r == 0.5 ? tinf : tret;

    ret = x < 0x1.0p+52 ? tret : ret;

    return ret;
}

