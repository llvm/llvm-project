/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE2(fma)(float2 a, float2 b, float2 c)
{
    return BUILTIN_FMA_2F32(a, b, c);
}

CONSTATTR float
MATH_MANGLE(fma)(float a, float b, float c)
{
    return BUILTIN_FMA_F32(a, b, c);
}

CONSTATTR float
MATH_MANGLE(fma_rte)(float a, float b, float c)
{
    return BUILTIN_FMA_F32(a, b, c);
}

#pragma STDC FENV_ACCESS ON

#define GEN(LN,RM) \
CONSTATTR float \
MATH_MANGLE(LN)(float a, float b, float c) \
{ \
    BUILTIN_SETROUND_F32(RM); \
    float ret = BUILTIN_FMA_F32(a, b, c); \
    BUILTIN_SETROUND_F32(ROUND_RTE); \
    return ret; \
}

GEN(fma_rtn, ROUND_RTN)
GEN(fma_rtp, ROUND_RTP)
GEN(fma_rtz, ROUND_RTZ)

