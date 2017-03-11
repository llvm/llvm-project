/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half2
MATH_MANGLE2(modf)(half2 x, __private half2 *iptr)
{
    half2 tx = BUILTIN_TRUNC_2F16(x);
    half2 ret = x - tx;
    ret.lo = BUILTIN_CLASS_F16(x.lo, CLASS_PINF|CLASS_NINF) ? 0.0h : ret.lo;
    ret.hi = BUILTIN_CLASS_F16(x.hi, CLASS_PINF|CLASS_NINF) ? 0.0h : ret.hi;
    *iptr = tx;
    return BUILTIN_COPYSIGN_2F16(ret, x);
}

INLINEATTR half
MATH_MANGLE(modf)(half x, __private half *iptr)
{
    half tx = BUILTIN_TRUNC_F16(x);
    half ret = x - tx;
    ret = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) ? 0.0h : ret;
    *iptr = tx;
    return BUILTIN_COPYSIGN_F16(ret, x);
}

