/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(pred)(float x)
{
    int ix = AS_INT(x) + (x > 0.0f ? -1 : 1);
    float y = x == 0.0f ? -0x1p-149f : AS_FLOAT(ix);
    return BUILTIN_ISNAN_F32(x) || x == NINF_F32 ? x : y;
}

