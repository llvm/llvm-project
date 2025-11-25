/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(nan)(uint nancode)
{
    return AS_FLOAT(QNANBITPATT_SP32 | (nancode & 0xfffff));
}

