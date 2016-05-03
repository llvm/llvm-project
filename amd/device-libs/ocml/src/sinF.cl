
#include "mathF.h"
#include "trigredF.h"

INLINEATTR float
MATH_MANGLE(sin)(float x)
{
    int ix = as_int(x);
    int ax = ix & 0x7fffffff;
    float dx = as_float(ax);

#if defined EXTRA_PRECISION
    float r0, r1;
    int regn = MATH_PRIVATE(trigred)(&r0, &r1, dx);

    float cc;
    float ss = MATH_PRIVATE(sincosred2)(r0, r1, &cc);
#else
    float r;
    int regn = MATH_PRIVATE(trigred)(&r, dx);

    float cc;
    float ss = MATH_PRIVATE(sincosred)(r, &cc);
#endif

    float s = (regn & 1) != 0 ? cc : ss;
    s = as_float(as_int(s) ^ ((regn > 1) << 31) ^ (ix ^ ax));

    if (!FINITE_ONLY_OPT()) {
        s = ax >= PINFBITPATT_SP32 ? as_float(QNANBITPATT_SP32) : s;
    }

    return s;
}

