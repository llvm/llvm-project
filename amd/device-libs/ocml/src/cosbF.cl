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

float
MATH_PRIVATE(cosb)(float x, int n, float p)
{
    struct redret r = MATH_PRIVATE(trigred)(x);
    bool b = r.hi < p;
    r.i = (r.i - b - n) & 3;

#if defined EXTRA_PRECISION
    float ph = AS_FLOAT(0xbf490fdb ^ (b ? 0x80000000 : 0));
    float pl = AS_FLOAT(0x32bbbd2e ^ (b ? 0x80000000 : 0));

    float sh, sl;

    FDIF2(ph, p, ph, sl);
    pl += sl;
    FSUM2(ph, pl, ph, pl);

    FSUM2(ph, r.hi, sh, sl);
    sl += pl + r.lo;
    FSUM2(sh, sl, sh, sl);

    struct scret sc = MATH_PRIVATE(sincosred2)(sh, sl);
#else
    r.hi = r.hi - p + AS_FLOAT(0xbf490fdb ^ (b ? 0x80000000 : 0));

    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
#endif
    sc.s = -sc.s;

    float c =  (r.i & 1) != 0 ? sc.s : sc.c;
    c = AS_FLOAT(AS_INT(c) ^ (r.i > 1 ? 0x80000000 : 0));
    return c;
}

