/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(rsqrt)(double x)
{
    double y0 = BUILTIN_AMDGPU_RSQRT_F64(x);
    double e = MATH_MAD(-y0 * (x == PINF_F64 || x == 0.0 ? y0 : x), y0, 1.0);
    return MATH_MAD(y0*e, MATH_MAD(e, 0.375, 0.5), y0);
}

