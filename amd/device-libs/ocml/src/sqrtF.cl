/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(sqrt)(float x)
{
    if (CORRECTLY_ROUNDED_SQRT32()) {
        return MATH_SQRT(x);
    } else {
        return MATH_FAST_SQRT(x);
    }
}

#if defined ENABLE_ROUNDED
#if defined HSAIL_BUILD

#define GEN(NAME,ROUND)\
CONSTATTR INLINEATTR float \
MATH_MANGLE(NAME)(float x) \
{ \
    float ret; \
    if (DAZ_OPT()) { \
        ret = BUILTIN_FULL_UNARY(fsqrtf, true, ROUND, x); \
    } else { \
        ret = BUILTIN_FULL_UNARY(fsqrtf, false, ROUND, x); \
    } \
    return ret; \
}

GEN(sqrt_rte, ROUND_TO_NEAREST_EVEN)
GEN(sqrt_rtp, ROUND_TO_POSINF)
GEN(sqrt_rtn, ROUND_TO_NEGINF)
GEN(sqrt_rtz, ROUND_TO_ZERO)

#endif // HSAIL_BUILD
#endif // ENABLE_ROUNDED

