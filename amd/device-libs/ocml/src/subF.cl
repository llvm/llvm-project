/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define GEN(LN,UN) \
CONSTATTR float \
MATH_MANGLE(LN)(float x, float y) \
{ \
    return BUILTIN_##UN##_F32(x, y); \
}

// GEN(sub_rte,SUB_RTE)
// GEN(sub_rtn,SUB_RTN)
// GEN(sub_rtp,SUB_RTP)
// GEN(sub_rtz,SUB_RTZ)

