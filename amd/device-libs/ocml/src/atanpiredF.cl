/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_PRIVATE(atanpired)(float v)
{
    float t = v * v;
    float z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  0x1.ccf836p-11f, -0x1.4761e4p-8f), 0x1.b6662ep-7f), -0x1.8423b4p-6f),
                  0x1.149cb4p-5f), -0x1.721cccp-5f), 0x1.04a466p-4f), -0x1.b2981cp-4f),
                  0x1.45f306p-2f);
    return v * z;
}

