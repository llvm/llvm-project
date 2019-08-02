/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

REQUIRES_16BIT_INSTS CONSTATTR short2
MATH_MANGLE2(isinf)(half2 x)
{
    return (short2)
        (BUILTIN_ISINF_F16(x.lo) ? (short)-1 : (short)0,
         BUILTIN_ISINF_F16(x.hi) ? (short)-1 : (short)0);
}

REQUIRES_16BIT_INSTS CONSTATTR int
MATH_MANGLE(isinf)(half x)
{
    return BUILTIN_ISINF_F16(x);
}

