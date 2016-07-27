/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(nan)(ulong nancode)
{
    return AS_DOUBLE((nancode & MANTBITS_DP64) | QNANBITPATT_DP64);
}

