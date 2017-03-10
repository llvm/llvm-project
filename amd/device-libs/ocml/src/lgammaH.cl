/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(lgamma)

INLINEATTR half
MATH_MANGLE(lgamma)(half x)
{
    int s;
    return MATH_MANGLE(lgamma_r)(x, &s);
}

