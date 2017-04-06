/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(logb)(float x)
{
    float ret = (float)(BUILTIN_FREXP_EXP_F32(x) - 1);

    if (!FINITE_ONLY_OPT()) {
        int ax = AS_INT(x) & EXSIGNBIT_SP32;
        ret = ax >= PINFBITPATT_SP32 ? AS_FLOAT(ax) : ret;
        ret = x == 0.0f ? AS_FLOAT(NINFBITPATT_SP32) : ret;
    }

    return ret;
}

