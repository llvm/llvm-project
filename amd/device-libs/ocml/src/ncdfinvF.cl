/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(ncdfinv)(float x)
{
    return -0x1.6a09e6p+0f * MATH_MANGLE(erfcinv)(x + x);
}

