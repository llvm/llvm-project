/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

#define GEN(LN,UN) \
CONSTATTR half \
MATH_MANGLE(LN)(half x, half y) \
{ \
    return BUILTIN_##UN##_F16(x, y); \
}

// GEN(div_rte,DIV_RTE)
// GEN(div_rtn,DIV_RTN)
// GEN(div_rtp,DIV_RTP)
// GEN(div_rtz,DIV_RTZ)

