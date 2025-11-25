/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_PRIVATE(bp1)(float t)
{
    return
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
            0x1.0214cep+1f, -0x1.7a54cap-2f), 0x1.4ffefep-3f), -0x1.800000p-2f);
}

