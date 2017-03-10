/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(fma)(half2 a, half2 b, half2 c)
{
    return BUILTIN_FMA_2F16(a, b, c);
}

CONSTATTR half
MATH_MANGLE(fma)(half a, half b, half c)
{
    return BUILTIN_FMA_F16(a, b, c);
}

#if defined ENABLE_ROUNDED
#if defined HSAIL_BUILD

#define GEN(NAME,ROUND)\
CONSTATTR INLINEATTR half \
MATH_MANGLE(NAME)(half a, half b, half c) \
{ \
    return BUILTIN_FULL_TERNARY(ffmah, false, ROUND, a, b, c); \
}

GEN(fma_rte, ROUND_TO_NEAREST_EVEN)
GEN(fma_rtp, ROUND_TO_POSINF)
GEN(fma_rtn, ROUND_TO_NEGINF)
GEN(fma_rtz, ROUND_TO_ZERO)

#endif // HSAIL_BUILD
#endif // ENABLE_ROUNDED

