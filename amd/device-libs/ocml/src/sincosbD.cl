
#include "mathD.h"
#include "trigredD.h"

// Allow H,L to be the same as A,B
#define FSUM2(A, B, H, L) \
    do { \
        double __a = A; \
        double __b = B; \
        double __s = __a + __b; \
        double __t = __b - (__s - __a); \
        H = __s; \
        L = __t; \
    } while (0)

// sincos for bessel functions j0, j1, y0, y1
// the argument must be adjusted by -pi/4 (n=0) or -3pi/4 (n=1)
INLINEATTR double
MATH_PRIVATE(sincosb)(double x, int n, __private double * cp)
{
    double y = BUILTIN_ABS_F64(x);

    double r0, r1;
    int regn = MATH_PRIVATE(trigred)(&r0, &r1, y);

    // adjust reduced argument by by -pi/4 (n=0) or -3pi/4 (n=1)
    regn = (regn - (r0 < 0.0) - n) & 3;

    const double piby4h = 0x1.921fb54442d18p-1;
    const double piby4t = 0x1.1a62633145c07p-55;
    double ph = r0 < 0.0f ? piby4h : -piby4h;
    double pt = r0 < 0.0f ? piby4t : -piby4t;
    double rh, rt, sh, st;
    FSUM2(ph, r0, rh, rt);
    FSUM2(pt, r1, sh, st);
    rt += sh;
    FSUM2(rh, rt, rh, rt);
    rt += st;
    FSUM2(rh, rt, r0, r1);

    double cc;
    double ss = MATH_PRIVATE(sincosred2)(r0, r1, &cc);

    int flip = regn > 1 ? (int)0x80000000 : 0;

    int2 s = AS_INT2((regn & 1) != 0 ? cc : ss);
    s.hi ^= flip ^ (x < 0.0 ? (int)0x80000000 : 0);
    ss = -ss;
    int2 c = AS_INT2(regn & 1 ? ss : cc);
    c.hi ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool xgeinf = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF);
        s = xgeinf ? AS_INT2(QNANBITPATT_DP64) : s;
        c = xgeinf ? AS_INT2(QNANBITPATT_DP64) : c;
    }

    *cp = AS_DOUBLE(c);
    return AS_DOUBLE(s);
}

