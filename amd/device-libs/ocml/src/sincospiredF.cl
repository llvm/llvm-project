/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

INLINEATTR float
MATH_PRIVATE(sincospired)(float x, __private float *cp)
{

    float t = x * x;

    float sx = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   0x1.eb5482p-3f, -0x1.3e497cp-1f), 0x1.468e6cp+1f), -0x1.4abc1cp+2f);
    sx = x * t * sx;
    sx = MATH_MAD(x, 0x1.921fb6p+1f, sx);

    float cx = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                   0x1.97ca88p-5f, 0x1.c85d3ap-3f), -0x1.55a3b4p+0f), 0x1.03c1a6p+2f),
                   -0x1.3bd3ccp+2f);
    cx = MATH_MAD(t, cx, 1.0f);

    *cp = cx;
    return sx;
}

