/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_PRIVATE(atanred)(float v)
{
    float t = v * v;
    float z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  0x1.5a54bp-9f, -0x1.f4b218p-7f), 0x1.53f67ep-5f), -0x1.2fa9aep-4f),
                  0x1.b26364p-4f), -0x1.22c1ccp-3f), 0x1.99717ep-3f), -0x1.5554c4p-2f);

    z = MATH_MAD(v, t*z, v);
    return z;
}

