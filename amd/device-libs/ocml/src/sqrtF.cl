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
  return __builtin_elementwise_sqrt(x);
}

#define GEN(LN,UN) \
CONSTATTR float \
MATH_MANGLE(LN)(float x) \
{ \
  return __builtin_elementwise_sqrt(x); \
}

// GEN(sqrt_rte,SQRT_RTE)
// GEN(sqrt_rtn,SQRT_RTN)
// GEN(sqrt_rtp,SQRT_RTP)
// GEN(sqrt_rtz,SQRT_RTZ)

