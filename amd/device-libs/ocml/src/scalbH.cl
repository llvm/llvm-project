/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(scalb)

CONSTATTR half
MATH_MANGLE(scalb)(half x, half y)
{
    half t = BUILTIN_MIN_F16(BUILTIN_MAX_F16(y, -0x1.0p+6h), 0x1.0p+6h);
    half ret = MATH_MANGLE(ldexp)(x, (int)BUILTIN_RINT_F16(t));

    if (!FINITE_ONLY_OPT()) {
        ret = (BUILTIN_ISNAN_F16(x) | BUILTIN_ISNAN_F16(y)) ? AS_HALF((short)QNANBITPATT_HP16) : ret;
        ret = (BUILTIN_CLASS_F16(x, CLASS_NZER|CLASS_PZER) & BUILTIN_CLASS_F16(y, CLASS_PINF)) ? AS_HALF((short)QNANBITPATT_HP16) : ret;
        ret = (BUILTIN_ISINF_F16(x) & BUILTIN_CLASS_F16(y, CLASS_NINF)) ? AS_HALF((short)QNANBITPATT_HP16) : ret;
    }

    return ret;
}

