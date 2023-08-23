/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_PRIVATE(atanpired)(half v)
{
    half t = v * v;
    half z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                 -0x1.ef4p-7h, 0x1.a44p-5h), -0x1.ac8p-4h), 0x1.46p-2h);
    return v * z;
}

