/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(rsqrt)

CONSTATTR INLINEATTR half
MATH_MANGLE(rsqrt)(half x)
{
    return BUILTIN_RSQRT_F16(x);
}

