/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(nan)(ushort nancode)
{
    ushort h = (ushort)QNANBITPATT_HP16 | (nancode & (ushort)0x01ff);
    return AS_HALF(h);
}

