/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR short2
MATH_MANGLE2(isnormal)(half2 x)
{
    return (short2)
        (BUILTIN_CLASS_F16(x.lo, CLASS_PNOR|CLASS_NNOR) ? (short)-1 : (short)0,
         BUILTIN_CLASS_F16(x.hi, CLASS_PNOR|CLASS_NNOR) ? (short)-1 : (short)0);
}

CONSTATTR INLINEATTR int
MATH_MANGLE(isnormal)(half x)
{
    return BUILTIN_CLASS_F16(x, CLASS_PNOR|CLASS_NNOR);
}

