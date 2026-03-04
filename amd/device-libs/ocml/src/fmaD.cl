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

CONSTATTR double
MATH_MANGLE(fma_rte)(double a, double b, double c)
{
    return BUILTIN_FMA_F64(a, b, c);
}

#pragma STDC FENV_ACCESS ON

#define GEN(LN,RM) \
CONSTATTR double \
MATH_MANGLE(LN)(double a, double b, double c) \
{ \
    BUILTIN_SETROUND_F16F64(RM); \
    double ret = BUILTIN_FMA_F64(a, b, c); \
    BUILTIN_SETROUND_F16F64(ROUND_RTE); \
    return ret; \
}

GEN(fma_rtn, ROUND_RTN)
GEN(fma_rtp, ROUND_RTP)
GEN(fma_rtz, ROUND_RTZ)

