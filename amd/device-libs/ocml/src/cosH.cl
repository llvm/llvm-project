
#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_MANGLE(cos)(half x)
{
    x = BUILTIN_ABS_F16(x);

    half r;
    int regn = MATH_PRIVATE(trigred)(&r, x);

    half cc;
    half ss = -MATH_PRIVATE(sincosred)(r, &cc);

    half c =  (regn & 1) != 0 ? ss : cc;
    half nc = -c;
    c = regn > 1 ? nc : c;

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? as_half((short)QNANBITPATT_HP16) : c;
    }

    return c;
}

