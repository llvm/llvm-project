
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(fdim)(double x, double y)
{
    long d = as_long(x - y);
    if (!FINITE_ONLY_OPT()) {
        long n = MATH_MANGLE(isnan)(x) | MATH_MANGLE(isnan)(y) ? QNANBITPATT_DP64 : 0L;
        return as_double(x > y ? d : n);
    } else {
	return as_double(x > y ? d : 0L);
    }
}

