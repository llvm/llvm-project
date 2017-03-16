/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_PRIVATE(atanred)(half v)
{
    half t = v * v;
    half z = MATH_MAD(t, MATH_MAD(t, -0x1.788p-5h, 0x1.44cp-3h), -0x1.4f4p-2h);
    z = MATH_MAD(v, t*z, v);
    return z;
}

