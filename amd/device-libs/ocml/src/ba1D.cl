/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(ba1)(double t)
{
    return
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
            -0x1.7940a06621145p+20, 0x1.591fb68428bafp+16), -0x1.996552a8bafb0p+11), 0x1.0795578cd8c93p+7),
            -0x1.ef38364596b5ap+2), 0x1.9c4fa465744c7p-1), -0x1.8bffffc3937c1p-3), 0x1.7ffffffffc240p-3),
            0x1.0000000000000p+0);
}

