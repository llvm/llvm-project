/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(bp0)(float t)
{
    return
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
            -0x1.5ec5e6p+0f, 0x1.aafb08p-3f), -0x1.0aa926p-4f), 0x1.000000p-3f);
}

