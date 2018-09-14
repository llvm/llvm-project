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

// GEN(add_rte,ADD_RTE)
// GEN(add_rtn,ADD_RTN)
// GEN(add_rtp,ADD_RTP)
// GEN(add_rtz,ADD_RTZ)

