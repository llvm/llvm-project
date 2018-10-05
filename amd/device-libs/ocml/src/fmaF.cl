/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "mathF.h"

CONSTATTR float
MATH_MANGLE(fma)(float a, float b, float c)
{
    return BUILTIN_FMA_F32(a, b, c);
}

#define GEN(LN,UN) \
CONSTATTR float \
MATH_MANGLE(LN)(float a, float b, float c) \
{ \
    return BUILTIN_##UN##_F32(a, b, c); \
}

// GEN(fma_rte,FMA_RTE)
// GEN(fma_rtn,FMA_RTN)
// GEN(fma_rtp,FMA_RTP)
// GEN(fma_rtz,FMA_RTZ)

