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
    double e = MATH_MAD(-x*y0, y0, 1.0);
    double y1 = MATH_MAD(y0*e, MATH_MAD(e, 0.375, 0.5), y0);
    return BUILTIN_CLASS_F64(y0, CLASS_PSUB|CLASS_PNOR) ? y1 : y0;
}

