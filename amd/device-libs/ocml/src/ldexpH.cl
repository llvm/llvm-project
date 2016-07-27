/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(ldexp)(half x, int n)
{
    if (AMD_OPT()) {
        return BUILTIN_FLDEXP_F16(x, n);
    } else {
        uint ux = (uint)AS_USHORT(x) & EXSIGNBIT_HP16;
        int e = (ux >> EXPSHIFTBITS_HP16) - EXPBIAS_HP16;
        int es = 7 - (int)MATH_CLZI(ux);
        uint m = ux & MANTBITS_HP16;
        uint ms = ux << (-14 - es);
        m = e == -EXPBIAS_HP16 ? ms : m;
        e = e == -EXPBIAS_HP16 ? es : e;
        n = BUILTIN_MIN_S32(BUILTIN_MAX_S32(n, -64), 64);
        int en = (short)BUILTIN_MIN_S32(BUILTIN_MAX_S32(e + n, -28), 19);
        int enh = en >> 1;
        half ret = AS_HALF((ushort)(((uint)AS_USHORT(x) ^ ux) | (EXPBIAS_HP16 << EXPSHIFTBITS_HP16) | m));
        ret *= AS_HALF((ushort)((EXPBIAS_HP16 + enh) << EXPSHIFTBITS_HP16));
        ret *= AS_HALF((ushort)((EXPBIAS_HP16 + (en - enh)) << EXPSHIFTBITS_HP16));
        ret = BUILTIN_CLASS_F16(x, CLASS_PZER|CLASS_NZER|CLASS_PINF|CLASS_NINF|CLASS_QNAN|CLASS_SNAN) ? x : ret;
        return ret;
    }
}

