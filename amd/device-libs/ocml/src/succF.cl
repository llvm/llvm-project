/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(succ)(float x)
{
    int y = AS_INT(x + 0.0f);
    int ix = y + (y >= 0 ? 1 : -1);

    float fx = AS_FLOAT(ix);
    if (DAZ_OPT())
      fx = x == 0.0f ? FLT_MIN : fx;

    return BUILTIN_ISNAN_F32(x) || x == PINF_F32 ? x : fx;
}
