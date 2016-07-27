/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(fma)(double a, double b, double c)
{
    return BUILTIN_FMA_F64(a, b, c);
}

#if defined ENABLE_ROUNDED
#if defined HSAIL_BUILD

#define GEN(NAME,ROUND)\
CONSTATTR INLINEATTR double \
MATH_MANGLE(NAME)(double a, double b, double c) \
{ \
    return BUILTIN_FULL_TERNARY(ffma, false, ROUND, a, b, c); \
}

GEN(fma_rte, ROUND_TO_NEAREST_EVEN)
GEN(fma_rtp, ROUND_TO_POSINF)
GEN(fma_rtn, ROUND_TO_NEGINF)
GEN(fma_rtz, ROUND_TO_ZERO)

#endif // HSAIL_BUILD
#endif // ENABLE_ROUNDED

