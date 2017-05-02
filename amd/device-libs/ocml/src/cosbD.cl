/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

#define FSUM2(A, B, H, L) \
    do { \
        double __s = A + B; \
        double __t = B - (__s - A); \
        H = __s; \
        L = __t; \
    } while (0)

#define FDIF2(A, B, H, L) \
    do { \
        double __d = A - B; \
        double __e = (A - __d) - B; \
        H = __d; \
        L = __e; \
    } while (0)

INLINEATTR double
MATH_PRIVATE(cosb)(double x, int n, double p)
{
    double ph, pl, rh, rl, sh, sl;
    int i = MATH_PRIVATE(trigred)(&rh, &rl, x);
    bool b = rh < p;
    i = (i - b - n) & 3;

    // This is a properly signed extra precise pi/4
    ph = AS_DOUBLE((uint2)(0x54442d18, 0xbfe921fb ^ (b ? 0x80000000 : 0)));
    pl = AS_DOUBLE((uint2)(0x33145c07, 0xbc81a626 ^ (b ? 0x80000000 : 0)));

    FDIF2(ph, p, ph, sl);
    pl += sl;
    FSUM2(ph, pl, ph, pl);

    FSUM2(ph, rh, sh, sl);
    sl += pl + rl;
    FSUM2(sh, sl, sh, sl);

    double cc;
    double ss = -MATH_PRIVATE(sincosred2)(sh, sl, &cc);

    int2 c = AS_INT2((i & 1) != 0 ? ss : cc);
    c.hi ^= i > 1 ? 0x80000000 : 0;

    return AS_DOUBLE(c);
}

