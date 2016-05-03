
#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_PRIVATE(sincosred)(half x, __private half *cp)
{
    const half c0 =  0x1.55554ap-5h;
    const half c1 = -0x1.6c0c2cp-10h;
    const half c2 =  0x1.99ebdap-16h;

    const half s0 = -0x1.555544p-3h;
    const half s1 =  0x1.11072ep-7h;
    const half s2 = -0x1.994430p-13h;

    half x2 = x*x;
    half c = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, c1, c0), -0.5h), 1.0h);
    half s = MATH_MAD(x, x2*MATH_MAD(x2, s1, s0), x);

    *cp = c;
    return s;
}

