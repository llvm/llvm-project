/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(logb)

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(logb)(half x)
{
    half ret = (half)(BUILTIN_FREXP_EXP_F16(x) - (short)1);

    if (!FINITE_ONLY_OPT()) {
        half ax = BUILTIN_ABS_F16(x);
        ret = BUILTIN_ISFINITE_F16(ax) ? ret : ax;
        ret = x == 0.0h ? NINF_F16 : ret;
    }

    return ret;
}

