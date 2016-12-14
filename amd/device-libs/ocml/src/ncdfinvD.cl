/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

INLINEATTR PUREATTR double
MATH_MANGLE(ncdfinv)(double x)
{
    return -0x1.6a09e667f3bcdp+0 * MATH_MANGLE(erfcinv)(x + x);
}

