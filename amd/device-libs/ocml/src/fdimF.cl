
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(fdim)(float x, float y)
{
    if (!FINITE_ONLY_OPT()) {
        int n = -(MATH_MANGLE(isnan)(x) | MATH_MANGLE(isnan)(y)) & QNANBITPATT_SP32;
        int r = -(x > y) & AS_INT(x - y);
        return AS_FLOAT(n | r);
    } else {
	return AS_FLOAT(-(x > y) & AS_INT(x - y));
    }
}

