/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

#if defined ENABLE_ROUNDED
#if defined HSAIL_BUILD

#define GEN(NAME,ROUND)\
CONSTATTR INLINEATTR half \
MATH_MANGLE(NAME)(half x, half y) \
{ \
    return BUILTIN_FULL_BINARY(fdivh, false, ROUND, x, y); \
}

GEN(div_rte, ROUND_TO_NEAREST_EVEN)
GEN(div_rtp, ROUND_TO_POSINF)
GEN(div_rtn, ROUND_TO_NEGINF)
GEN(div_rtz, ROUND_TO_ZERO)

#endif // HSAIL_BUILD
#endif // ENABLE_ROUNDED

