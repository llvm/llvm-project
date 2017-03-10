/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR short2
MATH_MANGLE2(isfinite)(half2 x)
{
    return (short2)
        (BUILTIN_CLASS_F16(x.lo, CLASS_NNOR|CLASS_NSUB|CLASS_NZER|CLASS_PZER|CLASS_PSUB|CLASS_PNOR) ? (short)-1 : (short)0,
         BUILTIN_CLASS_F16(x.hi, CLASS_NNOR|CLASS_NSUB|CLASS_NZER|CLASS_PZER|CLASS_PSUB|CLASS_PNOR) ? (short)-1 : (short)0);
}

CONSTATTR INLINEATTR int
MATH_MANGLE(isfinite)(half x)
{
    return BUILTIN_CLASS_F16(x, CLASS_NNOR|CLASS_NSUB|CLASS_NZER|CLASS_PZER|CLASS_PSUB|CLASS_PNOR);
}

