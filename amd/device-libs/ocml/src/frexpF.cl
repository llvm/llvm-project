/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

INLINEATTR float
MATH_MANGLE(frexp)(float x, __private int *ep)
{
    int e = BUILTIN_FREXP_EXP_F32(x);
    float r = BUILTIN_FREXP_MANT_F32(x);
    bool c = BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF|CLASS_SNAN|CLASS_QNAN);
    *ep = c ? 0 : e;
    return c ? x : r;
}

