/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR short2
MATH_MANGLE2(isnan)(half2 x)
{
    return (short2)
        (BUILTIN_ISNAN_F16(x.lo) ? (short)-1 : (short)0,
         BUILTIN_ISNAN_F16(x.hi) ? (short)-1 : (short)0);
}

CONSTATTR int
MATH_MANGLE(isnan)(half x)
{
    return BUILTIN_ISNAN_F16(x);
}

