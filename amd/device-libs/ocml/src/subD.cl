/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(sub_rte)(double x, double y)
{
    return x - y;
}

#pragma STDC FENV_ACCESS ON

#define GEN(LN,RM) \
CONSTATTR double \
MATH_MANGLE(LN)(double x, double y) \
{ \
    BUILTIN_SETROUND_F16F64(RM); \
    double ret = x - y; \
    BUILTIN_SETROUND_F16F64(ROUND_RTE); \
    return ret; \
}

GEN(sub_rtn, ROUND_RTN)
GEN(sub_rtp, ROUND_RTP)
GEN(sub_rtz, ROUND_RTZ)

