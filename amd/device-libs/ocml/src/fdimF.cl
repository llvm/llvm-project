/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
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

