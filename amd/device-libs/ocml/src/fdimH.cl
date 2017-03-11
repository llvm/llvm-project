/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(fdim)

CONSTATTR INLINEATTR half
MATH_MANGLE(fdim)(half x, half y)
{
    if (!FINITE_ONLY_OPT()) {
        int n = -(MATH_MANGLE(isnan)(x) | MATH_MANGLE(isnan)(y)) & QNANBITPATT_HP16;
        int r = -(x > y) & (int)AS_USHORT(x - y);
        return AS_HALF((ushort)(n | r));
    } else {
	return AS_HALF((ushort)(-(x > y) & (int)AS_USHORT(x - y)));
    }
}

