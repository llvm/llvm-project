/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

// Allow H,L to be the same as A,B
#define FSUM2(A, B, H, L) \
    do { \
        double __s = A + B; \
        double __t = B - (__s - A); \
        H = __s; \
        L = __t; \
    } while (0)

#define SUM2(A, B, H, L) \
    do { \
        double __s = A + B; \
        double __aa = __s - B; \
        double __bb = __s - __aa; \
        double __da = A - __aa; \
        double __db = B - __bb; \
        double __t = __da + __db; \
        H = __s; \
        L = __t; \
    } while (0)

#define PROD2(A, B, H, L) \
    do { \
        double __p = A * B; \
        double __q = BUILTIN_FMA_F64(A, B, -__p); \
        H = __p; \
        L = __q; \
    } while (0)

#define EVALUATE(A, B2, B1, B0, F2, F1, F0) \
    do { \
        double __p2h, __p2l, __p1h, __p1l, __p0h, __p0l; \
        double __v1h, __v1l, __v2h, __v2l, __w2h, __w2l; \
        double __e0, __e1, __e2, __e3; \
        PROD2(B0, A, __p0h, __p0l); \
        PROD2(B1, A, __p1h, __p1l); \
        PROD2(B2, A, __p2h, __p2l); \
        SUM2(__p2l, __p1h, __v2h, __v2l); \
        SUM2(__p1l, __p0h, __v1h, __v1l); \
        SUM2(__v2l, __v1h, __w2h, __w2l); \
        __e3 = __p2h; \
        __e2 = __v2h; \
        __e1 = __w2h; \
        __e0 = __w2l + __v1l + __p0l; \
        FSUM2(__e3, __e2, __e3, __e2); \
        FSUM2(__e2, __e1, __e2, __e1); \
        FSUM2(__e1, __e0, __e1, __e0); \
        F2 = __e3; \
        F1 = __e2; \
        F0 = __e1; \
    } while(0)
    
CONSTATTR struct redret
MATH_PRIVATE(trigredlarge)(double x)
{
    // Scale x by relevant part of 2/pi
    double p2 = BUILTIN_TRIG_PREOP_F64(x, 0);
    double p1 = BUILTIN_TRIG_PREOP_F64(x, 1);
    double p0 = BUILTIN_TRIG_PREOP_F64(x, 2);

    x = x >= 0x1.0p+945 ? BUILTIN_FLDEXP_F64(x, -128) : x;

    double f2, f1, f0;
    EVALUATE(x, p2, p1, p0, f2, f1, f0);

    f2 = BUILTIN_FLDEXP_F64(BUILTIN_FRACTION_F64(BUILTIN_FLDEXP_F64(f2, -2)), 2);
    f2 += f2+f1 < 0.0 ? 4.0 : 0.0;

    int i = (int)(f2 + f1);
    f2 -= (double)i;

    FSUM2(f2, f1, f2, f1);
    FSUM2(f1, f0, f1, f0);

    int g = f2 >= 0.5;
    i += g;
    f2 -= g ? 1.0 : 0.0;

    FSUM2(f2, f1, f2, f1);

    const double pio2h  = 0x1.921fb54442d18p+0;
    const double pio2t  = 0x1.1a62633145c07p-54;

    double rh = f2 * pio2h;
    double rt = BUILTIN_FMA_F64(f1, pio2h, BUILTIN_FMA_F64(f2, pio2t, BUILTIN_FMA_F64(f2, pio2h, -rh)));

    FSUM2(rh, rt, rh, rt);

    struct redret ret;
    ret.hi = rh;
    ret.lo = rt;
    ret.i = i & 0x3;
    return ret;
}

