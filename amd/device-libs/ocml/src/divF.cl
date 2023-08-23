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

// GEN(div_rte,DIV_RTE)
// GEN(div_rtn,DIV_RTN)
// GEN(div_rtp,DIV_RTP)
// GEN(div_rtz,DIV_RTZ)

