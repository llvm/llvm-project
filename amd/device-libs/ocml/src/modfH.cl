/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS half2
MATH_MANGLE2(modf)(half2 x, __private half2 *iptr)
{
    half2 tx = BUILTIN_TRUNC_2F16(x);
    half2 ret = x - tx;
    ret.lo = BUILTIN_ISINF_F16(x.lo) ? 0.0h : ret.lo;
    ret.hi = BUILTIN_ISINF_F16(x.hi) ? 0.0h : ret.hi;
    *iptr = tx;
    return BUILTIN_COPYSIGN_2F16(ret, x);
}

REQUIRES_16BIT_INSTS half
MATH_MANGLE(modf)(half x, __private half *iptr)
{
    half tx = BUILTIN_TRUNC_F16(x);
    half ret = x - tx;
    ret = BUILTIN_ISINF_F16(x) ? 0.0h : ret;
    *iptr = tx;
    return BUILTIN_COPYSIGN_F16(ret, x);
}

