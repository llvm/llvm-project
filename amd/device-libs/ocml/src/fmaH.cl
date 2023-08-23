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

CONSTATTR half
MATH_MANGLE(fma_rte)(half a, half b, half c)
{
    return BUILTIN_FMA_F16(a, b, c);
}

#pragma STDC FENV_ACCESS ON

#define GEN(LN,RM) \
CONSTATTR half \
MATH_MANGLE(LN)(half a, half b, half c) \
{ \
    BUILTIN_SETROUND_F16F64(RM); \
    half ret = BUILTIN_FMA_F16(a, b, c); \
    BUILTIN_SETROUND_F16F64(ROUND_RTE); \
    return ret; \
}

GEN(fma_rtn, ROUND_RTN)
GEN(fma_rtp, ROUND_RTP)
GEN(fma_rtz, ROUND_RTZ)

