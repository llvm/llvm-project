/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(fdim)(float x, float y)
{
    return (x <= y && !BUILTIN_ISUNORDERED_F32(x, y)) ? 0.0f : (x - y);
}

