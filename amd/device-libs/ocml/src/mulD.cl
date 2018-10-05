/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define GEN(LN,UN) \
CONSTATTR double \
MATH_MANGLE(LN)(double x, double y) \
{ \
    return BUILTIN_##UN##_F64(x, y); \
}

// GEN(mul_rte,MUL_RTE)
// GEN(mul_rtn,MUL_RTN)
// GEN(mul_rtp,MUL_RTP)
// GEN(mul_rtz,MUL_RTZ)

