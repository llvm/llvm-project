/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

UGEN(tan)

half
MATH_MANGLE(tan)(half x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F16(x) ? QNAN_F16 : x;

    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    half t = MATH_PRIVATE(tanred)(r.hi, r.i & (short)1);

    t = AS_HALF((short)(AS_SHORT(t) ^ (AS_SHORT(x) & SIGNBIT_HP16)));

    return t;
}

