/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_PRIVATE(bp1)(double t)
{
    return
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
        MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
        MATH_MAD(t, 
            0x1.c22f653d3a76ep+28, -0x1.80a4d95ed3e8ep+24), 0x1.72f1d1f8cdd76p+19), -0x1.3ea4e96460ad7p+14),
            0x1.488dd98d9ab3ap+9), -0x1.e9ed612fa3b38p+4), 0x1.2f484fcab9ddap+1), -0x1.7bccccad443c0p-2),
            0x1.4ffffffffcbfap-3), -0x1.8000000000000p-2);
}

