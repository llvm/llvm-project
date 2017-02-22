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
    half z = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                 0x1.bc9bfep-6h, -0x1.926ee8p-4h), 0x1.8310dcp-3h), -0x1.546844p-2h);
    z = MATH_MAD(v, t*z, v);
    return z;
}

