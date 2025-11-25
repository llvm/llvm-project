/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_PRIVATE(atanred)(half v)
{
    half t = v * v;
    half z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 0x1.938p-6h, -0x1.7f4p-4h), 0x1.7dcp-3h), -0x1.54p-2);
    z = MATH_MAD(t, v*z, v);
    return z;
}

