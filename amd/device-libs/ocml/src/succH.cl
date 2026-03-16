/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(succ)(half x)
{
    short y = AS_SHORT(x + 0.0h);
    short ix = y + (y >= 0 ? (short)1 : (short)-1);
    return BUILTIN_ISNAN_F16(x) || x == PINF_F16 ? x : AS_HALF(ix);
}

