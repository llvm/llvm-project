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

// GEN(sub_rte,SUB_RTE)
// GEN(sub_rtn,SUB_RTN)
// GEN(sub_rtp,SUB_RTP)
// GEN(sub_rtz,SUB_RTZ)

