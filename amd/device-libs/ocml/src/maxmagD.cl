/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(maxmag)(double x, double y)
{
#if 0
    long ix = AS_LONG(x);
    long iy = AS_LONG(y);
    long ax = ix & 0x7fffffffffffffffL;
    long ay = iy & 0x7fffffffffffffffL;
    ax |= -(ax > 0x7ff0000000000000L);
    ay |= -(ay > 0x7ff0000000000000L);
    return AS_DOUBLE((-(ax > ay) & ix) |
	             (-(ay > ax) & iy) |
		     (-(ax == ay) & ((ix & iy) | (ax & 0x0008000000000000L))));
#else
    x = BUILTIN_CANONICALIZE_F64(x);
    y = BUILTIN_CANONICALIZE_F64(y);
    double ret = BUILTIN_MAX_F64(x, y);
    double ax = BUILTIN_ABS_F64(x);
    double ay = BUILTIN_ABS_F64(y);
    ret = ax > ay ? x : ret;
    ret = ay > ax ? y : ret;
    return ret;
#endif
}

