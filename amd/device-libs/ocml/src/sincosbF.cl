/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

// Allow H,L to be the same as A,B
#define FSUM2(A, B, H, L) \
    do { \
        float __a = A; \
        float __b = B; \
        float __s = __a + __b; \
        float __t = __b - (__s - __a); \
        H = __s; \
        L = __t; \
    } while (0)

// sincos for bessel functions j0, j1, y0, y1
// the argument must be adjusted by -pi/4 (n=0) or -3pi/4 (n=1)
INLINEATTR float
MATH_PRIVATE(sincosb)(float x, int n, __private float *cp)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;
    float dx = AS_FLOAT(ax);

    const float piby4h = 0x1.921fb6p-1f;
    const float piby4t = -0x1.777a5cp-26f;

#if defined EXTRA_PRECISION
    float r0, r1;
    int regn = MATH_PRIVATE(trigred)(&r0, &r1, dx);

    // adjust reduced argument by by -pi/4 (n=0) or -3pi/4 (n=1)
    regn = (regn - (r0 < 0.0f) - n) & 3;
    float ph = r0 < 0.0f ? piby4h : -piby4h;
    float pt = r0 < 0.0f ? piby4t : -piby4t;
    float rh, rt, sh, st;
    FSUM2(ph, r0, rh, rt);
    FSUM2(pt, r1, sh, st);
    rt += sh;
    FSUM2(rh, rt, rh, rt);
    rt += st;
    FSUM2(rh, rt, r0, r1);

    float cc;
    float ss = MATH_PRIVATE(sincosred2)(r0, r1, &cc);
#else
    float r;
    int regn = MATH_PRIVATE(trigred)(&r, dx);

    regn = (regn - (r < 0.0f) - n) & 3;
    float ph = r < 0.0f ? piby4h : -piby4h;
    float pt = r < 0.0f ? piby4t : -piby4t;
    float rh, rt;
    FSUM2(ph, r, rh, rt);
    rt = pt + rt;
    r = rh + rt;

    float cc;
    float ss = MATH_PRIVATE(sincosred)(r, &cc);
#endif

    int flip = (regn > 1) << 31;
    float s = (regn & 1) != 0 ? cc : ss;
    s = AS_FLOAT(AS_INT(s) ^ flip ^ (ax ^ ix));
    ss = -ss;
    float c = (regn & 1) != 0 ? ss : cc;
    c = AS_FLOAT(AS_INT(c) ^ flip);

    if (!FINITE_ONLY_OPT()) {
        c = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : c;
        s = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : s;
    }

    *cp = c;
    return s;
}

