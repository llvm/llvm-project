/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

INLINEATTR float
MATH_MANGLE(modf)(float x, __private float *iptr)
{
    float tx = BUILTIN_TRUNC_F32(x);
    float ret = x - tx;
    ret = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF) ? 0.0f : ret;
    *iptr = tx;
    return BUILTIN_COPYSIGN_F32(ret, x);
}

