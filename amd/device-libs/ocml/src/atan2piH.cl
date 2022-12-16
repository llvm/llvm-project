
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

extern CONSTATTR half MATH_PRIVATE(atanpired)(half);

CONSTATTR BGEN(atan2pi)


REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(atan2pi)(half y, half x)
{
    half ax = BUILTIN_ABS_F16(x);
    half ay = BUILTIN_ABS_F16(y);
    half v = BUILTIN_MIN_F16(ax, ay);
    half u = BUILTIN_MAX_F16(ax, ay);

    half vbyu = MATH_DIV(v, u);

    half a = MATH_PRIVATE(atanpired)(vbyu);

    half at = 0.5h - a;
    a = ay > ax ? at : a;
    at = 1.0h - a;
    a = x < 0.0h ? at : a;

    at = AS_SHORT(x) < 0 ? 1.0h : 0.0h;
    a = y == 0.0h ? at : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        at = x < 0.0h ? 0.75h : 0.25h;
        a = (BUILTIN_ISINF_F16(x) & BUILTIN_ISINF_F16(y)) ?
            at : a;

        // x or y is NaN
        a = BUILTIN_ISUNORDERED_F16(x, y) ? QNAN_F16 : a;
    }

    return BUILTIN_COPYSIGN_F16(a, y);
}
