/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

__attribute__((always_inline)) short
MATH_PRIVATE(trigred)(__private half *r, half hx)
{
    const float twobypi = 0x1.45f306p-1f;
    const float pb2_a = 0x1.92p+0f;
    const float pb2_b = 0x1.fap-12f;
    const float pb2_c = 0x1.54442ep-20f;

    float x = (float)hx;
    float fn = BUILTIN_RINT_F32(x * twobypi);

    *r = (half)BUILTIN_MAD_F32(fn, -pb2_c, BUILTIN_MAD_F32(fn, -pb2_b, BUILTIN_MAD_F32(fn, -pb2_a, x)));

    return (int)fn & 0x3;
}

