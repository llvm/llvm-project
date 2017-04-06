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


// Outputs (C) must be different from A and Bs
#define EXPAND(A, B2, B1, B0, C5, C4, C3, C2, C1) \
    do { \
        double __p0h, __p1h, __p1l, __p2h, __p2l; \
        double __t1h, __t2h, __s1h; \
        __p0h = B0 * A; \
        PROD2(B1, A, __p1h, __p1l); \
        SUM2(__p1l, __p0h, __t1h, C1); \
        FSUM2(__p1h, __t1h, __s1h, C2); \
        PROD2(B2, A, __p2h, __p2l); \
        SUM2(__p2l, __s1h, __t2h, C3); \
        FSUM2(__p2h, __t2h, C5, C4); \
    } while (0)

#define SHIFT(C5, C4, C3, C2, C1) \
    do { \
        FSUM2(C5, C4, C5, C4); \
        FSUM2(C4, C3, C4, C3); \
        FSUM2(C3, C2, C3, C2); \
        C2 += C1; \
        FSUM2(C5, C4, C5, C4); \
        FSUM2(C4, C3, C4, C3); \
        C3 += C2; \
    } while (0)

int
MATH_PRIVATE(trigredlarge)(__private double *r, __private double *rr, double x)
{
    // Scale x by relevant part of 2/pi
    double p2 = BUILTIN_TRIG_PREOP_F64(x, 0);
    double p1 = BUILTIN_TRIG_PREOP_F64(x, 1);
    double p0 = BUILTIN_TRIG_PREOP_F64(x, 2);

    x = BUILTIN_FLDEXP_F64(x, x >= 0x1.0p+945 ? -128 : 0);

    double f2, f1, f0, c2, c1;
    EXPAND(x, p2, p1, p0, f2, f1, f0, c2, c1);
    SHIFT(f2, f1, f0, c2, c1);

    // Remove most significant integer bits
    f2 = BUILTIN_FLDEXP_F64(BUILTIN_FRACTION_F64(BUILTIN_FLDEXP_F64(f2, -16)), 16);

    // Don't let it become negative
    f2 += f2+f1 < 0.0 ? 0x1.0p+16 : 0.0;

    // Get integer part and strip off
    int i = (int)(f2 + f1);
    f2 -= (double)i;

    FSUM2(f2, f1, f2, f1);
    FSUM2(f1, f0, f1, f0);

    // if fraction >= 1/2, increment i and subtract 1 from f
    int g = f2 >= 0.5;
    i += g;
    f2 -= g ? 1.0 : 0.0;

    // Normalize
    FSUM2(f2, f1, f2, f1);

    // Scale by pi/2
    const double pio2h  = 0x1.921fb54442d18p+0;
    const double pio2t  = 0x1.1a62633145c07p-54;

    double rh = f2 * pio2h;
    double rt = BUILTIN_FMA_F64(f1, pio2h, BUILTIN_FMA_F64(f2, pio2t, BUILTIN_FMA_F64(f2, pio2h, -rh)));

    FSUM2(rh, rt, rh, rt);
    *r = rh;
    *rr = rt;

    return i & 0x3;
}

