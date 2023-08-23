/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(nan)(ushort2 nancode)
{
    ushort2 h = (ushort2)QNANBITPATT_HP16 | (nancode & (ushort2)0x01ff);
    return AS_HALF2(h);
}

CONSTATTR half
MATH_MANGLE(nan)(ushort nancode)
{
    ushort h = (ushort)QNANBITPATT_HP16 | (nancode & (ushort)0x01ff);
    return AS_HALF(h);
}

