/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(bp0)(double t)
{
    return
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
        MATH_MAD(t, 
            -0x1.91f780a4a989bp+28, 0x1.52a41923b70a7p+24), -0x1.40a5e31612a8dp+19), 0x1.0c9a0cbe3b3b8p+14),
            -0x1.0af76167fe583p+9), 0x1.778ea61b94139p+4), -0x1.a3581d1a82662p+0), 0x1.ad33330a1daf2p-3),
            -0x1.0aaaaaaaa7909p-4), 0x1.0000000000000p-3);
}

