/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"


CONSTATTR INLINEATTR half
MATH_MANGLE(fmax)(half x, half y)
{
    // The multiplications are for sNAN quieting
    return BUILTIN_MAX_F16(BUILTIN_CANONICALIZE_F16(x), BUILTIN_CANONICALIZE_F16(y));
}

