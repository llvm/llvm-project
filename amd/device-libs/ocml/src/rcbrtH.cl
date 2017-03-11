/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(rcbrt)

PUREATTR half
MATH_MANGLE(rcbrt)(half x)
{
    if (AMD_OPT()) {
        half ret = (half)BUILTIN_EXP2_F32(-0x1.555556p-2f * BUILTIN_LOG2_F32((float)BUILTIN_ABS_F16(x)));
        if (!FINITE_ONLY_OPT()) {
            ret = x == 0.0h ? AS_HALF((short)PINFBITPATT_HP16) : ret;
            ret = BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF) ? 0.0h : ret;
            ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN) ? x : ret;
        }
        ret = BUILTIN_COPYSIGN_F16(ret, x);
        return ret;
    } else {
        return (half)MATH_UPMANGLE(rcbrt)((float)x);
    }
}

