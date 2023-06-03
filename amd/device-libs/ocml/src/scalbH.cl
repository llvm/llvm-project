/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(scalb)

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(scalb)(half x, half y)
{
    half t = BUILTIN_MIN_F16(BUILTIN_MAX_F16(y, -0x1.0p+6h), 0x1.0p+6h);
    half ret = MATH_MANGLE(ldexp)(x, (int)BUILTIN_RINT_F16(t));

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_ISUNORDERED_F16(x, y) ? QNAN_F16 : ret;
        ret = ((x == 0.0h) & (y == PINF_F16)) ? QNAN_F16 : ret;
        ret = (BUILTIN_ISINF_F16(x) & (y == NINF_F16)) ? QNAN_F16 : ret;
    }

    return ret;
}

