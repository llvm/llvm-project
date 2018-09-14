/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(fma)(half2 a, half2 b, half2 c)
{
    return BUILTIN_FMA_2F16(a, b, c);
}

CONSTATTR half
MATH_MANGLE(fma)(half a, half b, half c)
{
    return BUILTIN_FMA_F16(a, b, c);
}

#define GEN(LN,UN) \
CONSTATTR half \
MATH_MANGLE(LN)(half a, half b, half c) \
{ \
    return BUILTIN_##UN##_F16(a, b, c); \
}

// GEN(fma_rte,FMA_RTE)
// GEN(fma_rtn,FMA_RTN)
// GEN(fma_rtp,FMA_RTP)
// GEN(fma_rtz,FMA_RTZ)

