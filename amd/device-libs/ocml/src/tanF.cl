
#include "mathF.h"
#include "trigredF.h"

INLINEATTR float
MATH_MANGLE(tan)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;
    float dx = AS_FLOAT(ax);

#if defined EXTRA_PRECISION
    float r0, r1;
    int regn = MATH_PRIVATE(trigred)(&r0, &r1, dx);

    float t = MATH_PRIVATE(tanred)(r0 + r1, regn);
#else
    float r;
    int regn = MATH_PRIVATE(trigred)(&r, dx);

    float t = MATH_PRIVATE(tanred)(r, regn);
#endif

    t = AS_FLOAT(AS_INT(t) ^ (ix ^ ax));

    if (!FINITE_ONLY_OPT()) {
        t = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : t;
    }

    return t;
}

