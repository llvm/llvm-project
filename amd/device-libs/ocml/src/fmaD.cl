/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(fma)(double a, double b, double c)
{
    return BUILTIN_FMA_F64(a, b, c);
}

#define GEN(LN,UN) \
CONSTATTR double \
MATH_MANGLE(LN)(double a, double b, double c) \
{ \
    return BUILTIN_##UN##_F64(a, b, c); \
}

// GEN(fma_rte,FMA_RTE)
// GEN(fma_rtn,FMA_RTN)
// GEN(fma_rtp,FMA_RTP)
// GEN(fma_rtz,FMA_RTZ)

