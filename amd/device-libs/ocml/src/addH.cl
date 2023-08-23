/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(add_rte)(half x, half y)
{
    return x + y;
}

#pragma STDC FENV_ACCESS ON

#define GEN(LN,RM) \
CONSTATTR half \
MATH_MANGLE(LN)(half x, half y) \
{ \
    BUILTIN_SETROUND_F16F64(RM); \
    half ret = x + y; \
    BUILTIN_SETROUND_F16F64(ROUND_RTE); \
    return ret; \
}

GEN(add_rtn, ROUND_RTN)
GEN(add_rtp, ROUND_RTP)
GEN(add_rtz, ROUND_RTZ)

