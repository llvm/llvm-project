/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_PRIVATE(ba0)(double t)
{
    return
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
            0x1.44395cd7ac32cp+20, -0x1.25bf3abbee803p+16), 0x1.55a4a78625b0fp+11), -0x1.a826c7ea56321p+6),
            0x1.763253bbf53b6p+2), -0x1.15efaff948953p-1), 0x1.a7ffff967a1d4p-4), -0x1.fffffffff2868p-5),
            0x1.0000000000000p+0);
}

