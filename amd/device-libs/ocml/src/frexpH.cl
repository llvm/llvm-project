/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

INLINEATTR half
MATH_MANGLE(frexp)(half x, __private int *ep)
{
    if (AMD_OPT()) {
        int e = (int)BUILTIN_FREXP_EXP_F16(x);
        half r = BUILTIN_FREXP_MANT_F16(x);
        bool c = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF|CLASS_SNAN|CLASS_QNAN);
        *ep = c ? 0 : e;
        return c ? x : r;
    } else {
        int i = (int)AS_USHORT(x);
        int ai = i & EXSIGNBIT_HP16;
        bool d = ai > 0 & ai < IMPBIT_HP16;
        int s = (int)AS_USHORT(AS_HALF((ushort)(ONEEXPBITS_HP16 | ai)) - 1.0h);
        ai = d ? s : ai;
        int e = (ai >> 10) - (d ? 28 : 14);
        bool t = ai == 0 | e == 17;
        i = (i & SIGNBIT_HP16) | HALFEXPBITS_HP16 | (ai & MANTBITS_HP16);
        *ep = t ? 0 : e;
        return t ? x : AS_HALF((ushort)i);
    }
}

