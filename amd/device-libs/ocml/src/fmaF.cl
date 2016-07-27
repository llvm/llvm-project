/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(fma)(float a, float b, float c)
{
    return BUILTIN_FMA_F32(a, b, c);
}

#if defined ENABLE_ROUNDED
#if defined HSAIL_BUILD

#define GEN(NAME,ROUND)\
CONSTATTR INLINEATTR float \
MATH_MANGLE(NAME)(float a, float b, float c) \
{ \
    float ret; \
    if (DAZ_OPT()) { \
        ret = BUILTIN_FULL_TERNARY(ffmaf, true, ROUND, a, b, c); \
    } else { \
        ret = BUILTIN_FULL_TERNARY(ffmaf, false, ROUND, a, b, c); \
    } \
    return ret; \
}

GEN(fma_rte, ROUND_TO_NEAREST_EVEN)
GEN(fma_rtp, ROUND_TO_POSINF)
GEN(fma_rtn, ROUND_TO_NEGINF)
GEN(fma_rtz, ROUND_TO_ZERO)

#endif // HSAIL_BUILD
#endif // ENABLE_ROUNDED

