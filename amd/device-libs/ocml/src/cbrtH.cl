/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(cbrt)

CONSTATTR half
MATH_MANGLE(cbrt)(half x)
{
    half ret = (half)BUILTIN_EXP2_F32(0x1.555556p-2f * BUILTIN_LOG2_F32((float)BUILTIN_ABS_F16(x)));
    ret = BUILTIN_COPYSIGN_F16(ret, x);
    ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER) ? x : ret;
    return ret;
}

