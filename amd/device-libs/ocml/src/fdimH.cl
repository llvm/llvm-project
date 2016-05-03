
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(fdim)(half x, half y)
{
    if (!FINITE_ONLY_OPT()) {
        int n = -(MATH_MANGLE(isnan)(x) | MATH_MANGLE(isnan)(y)) & QNANBITPATT_HP16;
        int r = -(x > y) & (int)as_ushort(x - y);
        return as_half((ushort)(n | r));
    } else {
	return as_half((ushort)(-(x > y) & (int)as_ushort(x - y)));
    }
}

