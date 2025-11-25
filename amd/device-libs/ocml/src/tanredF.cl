/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR float
MATH_PRIVATE(tanred)(float x, int i)
{
    float s = x * x;

#if defined MORE_ACCURACY
    float p = s * MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s,
                  MATH_MAD(s,
                      0x1.33d5e6p-7f, 0x1.9697f8p-9f), 0x1.907be2p-6f), 0x1.b581ap-5f),
                      0x1.112e2p-3f), 0x1.5554dcp-2f);
#else
    float a = MATH_MAD(s, -0x1.19dba6p-6f, 0x1.8a8b0ep-2f);
    float b = MATH_MAD(s, MATH_MAD(s, 0x1.2e2900p-6f, -0x1.07266ep-1f), 0x1.27e84ap+0f);
    float p = s * MATH_FAST_DIV(a,b);
#endif

#if defined LESS_ACCURACY
    float t = MATH_MAD(p, x, x);
    float tr = -MATH_FAST_RCP(t);
#else
    float t = BUILTIN_FMA_F32(p, x, x);
    float tt = BUILTIN_FMA_F32(p, x, -(t - x));
    float tr = -MATH_FAST_RCP(t);
    float e = BUILTIN_FMA_F32(tt, tr, BUILTIN_FMA_F32(t, tr, 1.0f));
    tr = BUILTIN_FMA_F32(e, tr, tr);
#endif

    return i ? tr : t;
}

