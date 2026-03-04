/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half
MATH_MANGLE(pred)(half x)
{
    short ix = AS_SHORT(x) + (x > 0.0h ? (short)-1 : (short)1);
    half y = x == 0.0h ? -0x1p-24h : AS_HALF(ix);
    return BUILTIN_ISNAN_F16(x) || x == NINF_F16 ? x : y;
}

