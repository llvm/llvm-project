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

// GEN(div_rte,DIV_RTE)
// GEN(div_rtn,DIV_RTN)
// GEN(div_rtp,DIV_RTP)
// GEN(div_rtz,DIV_RTZ)

