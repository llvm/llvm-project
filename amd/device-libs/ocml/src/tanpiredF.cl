/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

CONSTATTR float
MATH_PRIVATE(tanpired)(float x, int i)
{
    float s = x * x;

    float t = MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, MATH_MAD(s, 
              MATH_MAD(s, MATH_MAD(s, 
                  0x1.7d2bd4p+16f, 0x1.a4d306p+12f), 0x1.435004p+11f), 0x1.4b6926p+9f),
                  0x1.451e22p+7f), 0x1.467a9cp+5f), 0x1.4abb6ap+3f);

    t = x * s * t;
    t = MATH_MAD(x, 0x1.921fb6p+1f, t);

    float tr = -MATH_RCP(t);

    return i ? tr : t;
}

