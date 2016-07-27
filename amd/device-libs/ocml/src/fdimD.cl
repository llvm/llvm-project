/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(fdim)(double x, double y)
{
    long d = AS_LONG(x - y);
    if (!FINITE_ONLY_OPT()) {
        long n = MATH_MANGLE(isnan)(x) | MATH_MANGLE(isnan)(y) ? QNANBITPATT_DP64 : 0L;
        return AS_DOUBLE(x > y ? d : n);
    } else {
	return AS_DOUBLE(x > y ? d : 0L);
    }
}

