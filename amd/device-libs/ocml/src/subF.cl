/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if defined ENABLE_ROUNDED
#if defined HSAIL_BUILD

#define GEN(NAME,ROUND)\
CONSTATTR INLINEATTR float \
MATH_MANGLE(NAME)(float x, float y) \
{ \
    float ret; \
    if (DAZ_OPT()) { \
        ret = BUILTIN_FULL_BINARY(fsubf, true, ROUND, x, y); \
    } else { \
        ret = BUILTIN_FULL_BINARY(fsubf, false, ROUND, x, y); \
    } \
    return ret; \
}

GEN(sub_rte, ROUND_TO_NEAREST_EVEN)
GEN(sub_rtp, ROUND_TO_POSINF)
GEN(sub_rtn, ROUND_TO_NEGINF)
GEN(sub_rtz, ROUND_TO_ZERO)

#endif // HSAIL_BUILD
#endif // ENABLE_ROUNDED

