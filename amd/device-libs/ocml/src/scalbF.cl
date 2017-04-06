/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(scalb)(float x, float y)
{
    float t = BUILTIN_CLAMP_F32(y, -0x1.0p+20f, 0x1.0p+20f);
    float ret = MATH_MANGLE(ldexp)(x, (int)BUILTIN_RINT_F32(t));

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN) | BUILTIN_CLASS_F32(y, CLASS_QNAN|CLASS_SNAN)) ?  AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = (BUILTIN_CLASS_F32(x, CLASS_NZER|CLASS_PZER) & BUILTIN_CLASS_F32(y, CLASS_PINF)) ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = (BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF) & BUILTIN_CLASS_F32(y, CLASS_NINF)) ? AS_FLOAT(QNANBITPATT_SP32) : ret;
    }

    return ret;
}

