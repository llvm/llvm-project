
#include "mathF.h"
#include "trigredF.h"

INLINEATTR float
MATH_MANGLE(sincos)(float x, __private float *cp)
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

    int flip = (regn > 1) << 31;
    float s = (regn & 1) != 0 ? cc : ss;
    s = as_float(as_int(s) ^ flip ^ (ax ^ ix));
    ss = -ss;
    float c = (regn & 1) != 0 ? ss : cc;
    c = as_float(as_int(c) ^ flip);

    if (!FINITE_ONLY_OPT()) {
        c = ax >= PINFBITPATT_SP32 ? as_float(QNANBITPATT_SP32) : c;
        s = ax >= PINFBITPATT_SP32 ? as_float(QNANBITPATT_SP32) : s;
    }

    *cp = c;
    return s;
}

