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

// GEN(add_rte,ADD_RTE)
// GEN(add_rtn,ADD_RTN)
// GEN(add_rtp,ADD_RTP)
// GEN(add_rtz,ADD_RTZ)

