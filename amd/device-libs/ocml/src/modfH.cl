/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half
MATH_MANGLE(modf)(half x, __private half *iptr)
{
    half tx = BUILTIN_TRUNC_F16(x);
    half ret = x - tx;
    ret = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) ? 0.0h : ret;
    *iptr = tx;
    return BUILTIN_COPYSIGN_F16(ret, x);
}

