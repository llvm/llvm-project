
#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_MANGLE(sincos)(half x, __private half *cp)
{
    half y = BUILTIN_ABS_F16(x);

    half r;
    int regn = MATH_PRIVATE(trigred)(&r, y);

    half cc;
    half ss = MATH_PRIVATE(sincosred)(r, &cc);

    bool flip = regn > 1;
    bool odd = (regn & 1) != 0;
    half s = odd ? cc : ss;
    half ns = -s;
    s = flip ^ (x < 0.0h) ? ns : s;
    ss = -ss;
    half c = odd ? ss : cc;
    half nc = -c;
    c = flip ? nc : c;

    if (!FINITE_ONLY_OPT()) {
        bool b = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF);
        c = b ? AS_HALF((short)QNANBITPATT_HP16) : c;
        s = b ? AS_HALF((short)QNANBITPATT_HP16) : s;
    }

    *cp = c;
    return s;
}

