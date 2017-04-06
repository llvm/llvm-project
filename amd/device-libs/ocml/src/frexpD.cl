/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

INLINEATTR double
MATH_MANGLE(frexp)(double x, __private int *ep)
{
    int e = BUILTIN_FREXP_EXP_F64(x);
    double r = BUILTIN_FREXP_MANT_F64(x);
    bool c = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF|CLASS_SNAN|CLASS_QNAN);
    *ep = c ? 0 : e;
    return c ? x : r;
}

