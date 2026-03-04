/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(sqrt)(double x)
{
    return MATH_SQRT(x);
}

#define GEN(LN,UN) \
CONSTATTR double \
MATH_MANGLE(LN)(double x) \
{ \
    return BUILTIN_##UN##_F64(x); \
}

// GEN(sqrt_rte,SQRT_RTE)
// GEN(sqrt_rtn,SQRT_RTN)
// GEN(sqrt_rtp,SQRT_RTP)
// GEN(sqrt_rtz,SQRT_RTZ)

