/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(ba0)(float t)
{
    return
        MATH_MAD(t, MATH_MAD(t, 
            0x1.92aeccp-4f, -0x1.ffe472p-5f), 0x1.000000p+0f);
}

