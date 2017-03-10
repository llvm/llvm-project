/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(maxmag)

CONSTATTR INLINEATTR half
MATH_MANGLE(maxmag)(half x, half y)
{
    x = BUILTIN_CANONICALIZE_F16(x);
    y = BUILTIN_CANONICALIZE_F16(y);
    half ret = BUILTIN_MAX_F16(x, y);
    half ax = BUILTIN_ABS_F16(x);
    half ay = BUILTIN_ABS_F16(y);
    ret = ax > ay ? x : ret;
    ret = ay > ax ? y : ret;
    return ret;
}

