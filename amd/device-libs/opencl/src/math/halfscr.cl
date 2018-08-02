/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"

__attribute__((always_inline)) float
__half_scr(float x, __private float *cp)
{
    float y = x * 0x1.45f306p-3f;
    *cp = __builtin_amdgcn_cosf(y);
    float s = __builtin_amdgcn_sinf(y);
    return fabs(x) < 0x1.0p-20f ? x : s;
}

