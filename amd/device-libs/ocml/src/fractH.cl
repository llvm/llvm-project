/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half
MATH_MANGLE(fract)(half x, __private half *ip)
{
    *ip = BUILTIN_FLOOR_F16(x);
    return  BUILTIN_FRACTION_F16(x);
}

