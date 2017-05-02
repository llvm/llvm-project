/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_PRIVATE(ba1)(float t)
{
    return
        MATH_MAD(t, MATH_MAD(t, 
            -0x1.7c0d46p-3f, 0x1.7ff5aap-3f), 0x1.000000p+0f);
}

