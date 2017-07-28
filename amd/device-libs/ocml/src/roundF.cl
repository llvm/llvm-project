/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(round)(float x)
{
    float t = BUILTIN_TRUNC_F32(x);
    float d = BUILTIN_ABS_F32(x - t);
    float o = BUILTIN_COPYSIGN_F32(1.0f, x);
    return t + (d >= 0.5f ? o : 0.0f);
}

