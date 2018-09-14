/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(sqrt)(float x)
{
    if (CORRECTLY_ROUNDED_SQRT32()) {
        return MATH_SQRT(x);
    } else {
        return MATH_FAST_SQRT(x);
    }
}

#define GEN(LN,UN) \
CONSTATTR float \
MATH_MANGLE(LN)(float x) \
{ \
    return BUILTIN_##UN##_F32(x); \
}

// GEN(sqrt_rte,SQRT_RTE)
// GEN(sqrt_rtn,SQRT_RTN)
// GEN(sqrt_rtp,SQRT_RTP)
// GEN(sqrt_rtz,SQRT_RTZ)

