/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

float
MATH_MANGLE(frexp)(float x, __private int *ep)
{
    return BUILTIN_FREXP_F32(x, ep);
}

