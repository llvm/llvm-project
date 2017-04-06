/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(rcbrt)

CONSTATTR half
MATH_MANGLE(rcbrt)(half x)
{
    half ret = (half)BUILTIN_EXP2_F32(-0x1.555556p-2f * BUILTIN_LOG2_F32((float)BUILTIN_ABS_F16(x)));

    half xi = MATH_FAST_RCP(x);
    ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_PZER|CLASS_NZER|CLASS_PINF|CLASS_NINF) ? xi : ret;

    return ret = BUILTIN_COPYSIGN_F16(ret, x);
}

