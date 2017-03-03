/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR INLINEATTR float
MATH_PRIVATE(tanred)(float x, int i)
{
    float s = x * x;

#if defined MORE_ACCURACY
    float t = MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s,
              MATH_MAD(s, MATH_MAD(s,
                  0x1.65e368p-8f, -0x1.334754p-9f), 0x1.a93cacp-7f), 0x1.4d80eap-6f),
                  0x1.bc8056p-5f), 0x1.1103bep-3f), 0x1.555578p-2f);
    t = MATH_MAD(x*s, t, x);
    float tr = -MATH_RCP(t);
#else
    float a = MATH_MAD(s, -0x1.19dba6p-6f, 0x1.8a8b0ep-2f);
    float b = MATH_MAD(s, MATH_MAD(s, 0x1.2e2900p-6f, -0x1.07266ep-1f), 0x1.27e84ap+0f);
    float t = MATH_MAD(x*s, MATH_FAST_DIV(a, b), x);
    float tr = -MATH_FAST_RCP(t);
#endif

    return i ? tr : t;
}

