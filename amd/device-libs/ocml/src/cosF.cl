/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

INLINEATTR float
MATH_MANGLE(cos)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;
    float dx = AS_FLOAT(ax);

#if defined EXTRA_PRECISION
    float r0, r1;
    int regn = MATH_PRIVATE(trigred)(&r0, &r1, dx);

    float cc;
    float ss = -MATH_PRIVATE(sincosred2)(r0, r1, &cc);
#else
    float r;
    int regn = MATH_PRIVATE(trigred)(&r, dx);

    float cc;
    float ss = -MATH_PRIVATE(sincosred)(r, &cc);
#endif

    float c =  (regn & 1) != 0 ? ss : cc;
    c = AS_FLOAT(AS_INT(c) ^ ((regn > 1) << 31));

    if (!FINITE_ONLY_OPT()) {
        c = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : c;
    }

    return c;
}

