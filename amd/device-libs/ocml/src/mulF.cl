/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(mul_rte)(float x, float y)
{
    return x * y;
}

#pragma STDC FENV_ACCESS ON

#define GEN(LN,RM) \
CONSTATTR float \
MATH_MANGLE(LN)(float x, float y) \
{ \
    BUILTIN_SETROUND_F32(RM); \
    float ret = x * y; \
    BUILTIN_SETROUND_F32(ROUND_RTE); \
    return ret; \
}

GEN(mul_rtn, ROUND_RTN)
GEN(mul_rtp, ROUND_RTP)
GEN(mul_rtz, ROUND_RTZ)

