/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

extern CONSTATTR half MATH_PRIVATE(atanred)(half);

CONSTATTR UGEN(atan)

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(atan)(half x)
{
    half v = BUILTIN_ABS_F16(x);
    bool g = v > 1.0h;

    half vi = MATH_FAST_RCP(v);
    v = g ? vi : v;

    half a = MATH_PRIVATE(atanred)(v);

    half y = MATH_MAD(0x1.ea8p-1h, 0x1.a3cp+0h, -a);
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F16(a, x);
}

