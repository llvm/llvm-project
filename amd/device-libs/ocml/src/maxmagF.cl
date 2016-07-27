/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(maxmag)(float x, float y)
{
#if 0
    int ix = AS_INT(x);
    int iy = AS_INT(y);
    int ax = ix & 0x7fffffff;
    int ay = iy & 0x7fffffff;
    ax |= -(ax > 0x7f800000);
    ay |= -(ay > 0x7f800000);
    return AS_FLOAT((-(ax > ay) & ix) |
	            (-(ay > ax) & iy) |
		    (-(ax == ay) & ((ix & iy) | (ax & 0x00400000))));
#else
    x = BUILTIN_CANONICALIZE_F32(x);
    y = BUILTIN_CANONICALIZE_F32(y);
    float ret = BUILTIN_MAX_F32(x, y);
    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    ret = ax > ay ? x : ret;
    ret = ay > ax ? y : ret;
    return ret;
#endif
}

