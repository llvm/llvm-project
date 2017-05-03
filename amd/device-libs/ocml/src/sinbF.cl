/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

#define FSUM2(A, B, H, L) \
    do { \
        float __s = A + B; \
        float __t = B - (__s - A); \
        H = __s; \
        L = __t; \
    } while (0)

#define FDIF2(A, B, H, L) \
    do { \
        float __d = A - B; \
        float __e = (A - __d) - B; \
        H = __d; \
        L = __e; \
    } while (0)

INLINEATTR float
MATH_PRIVATE(sinb)(float x, int n, float p)
{
#if defined EXTRA_PRECISION
    float ph, pl, rh, rl, sh, sl;
    int i = MATH_PRIVATE(trigred)(&rh, &rl, x);
    bool b = rh < p;
    i = (i - b - n) & 3;

    ph = AS_FLOAT(0xbf490fdb ^ (b ? 0x80000000 : 0));
    pl = AS_FLOAT(0x32bbbd2e ^ (b ? 0x80000000 : 0));

    FDIF2(ph, p, ph, sl);
    pl += sl;
    FSUM2(ph, pl, ph, pl);

    FSUM2(ph, rh, sh, sl);
    sl += pl + rl;
    FSUM2(sh, sl, sh, sl);

    float cc;
    float ss = MATH_PRIVATE(sincosred2)(sh, sl, &cc);
#else
    float r;
    int i = MATH_PRIVATE(trigred)(&r, x);
    bool b = r < p;
    i = (i - b - n) & 3;
    r = r - p + AS_FLOAT(0xbf490fdb ^ (b ? 0x80000000 : 0));

    float cc;
    float ss = MATH_PRIVATE(sincosred)(r, &cc);
#endif

    float s = (i & 1) != 0 ? cc : ss;
    s = AS_FLOAT(AS_INT(s) ^ (i > 1 ? 0x80000000 : 0));
    return s;
}

