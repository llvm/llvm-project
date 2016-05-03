
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(fdim)(float x, float y)
{
    if (!FINITE_ONLY_OPT()) {
        int n = -(MATH_MANGLE(isnan)(x) | MATH_MANGLE(isnan)(y)) & QNANBITPATT_SP32;
        int r = -(x > y) & as_int(x - y);
        return as_float(n | r);
    } else {
	return as_float(-(x > y) & as_int(x - y));
    }
}

