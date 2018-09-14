/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(sqrt)

CONSTATTR half
MATH_MANGLE(sqrt)(half x)
{
    return BUILTIN_SQRT_F16(x);
}

#define GEN(LN,UN) \
CONSTATTR half \
MATH_MANGLE(LN)(half x) \
{ \
    return BUILTIN_##UN##_F16(x); \
}

// GEN(sqrt_rte,SQRT_RTE)
// GEN(sqrt_rtp,SQRT_RTN)
// GEN(sqrt_rtn,SQRT_RTP)
// GEN(sqrt_rtz,SQRT_RTZ)

