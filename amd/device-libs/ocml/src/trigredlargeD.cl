
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

static double
get_twobypi_bits(int start, int scale)
{
    USE_TABLE(uint, twobypi, M64_PIBITS);

    int i = start >> 5;
    int b = start & 0x1f;
    uint w2 = twobypi[i];
    uint w1 = twobypi[i+1];
    uint w0 = twobypi[i+2];
    uint t;

    t = (w2 << b) | (w1 >> (32-b));
    w2 = b != 0 ? t : w2;

    t = (w1 << b) | (w0 >> (32-b));
    w1 = b != 0 ? t : w1;
    w1 &= 0xfffff800;

    int z = (int)MATH_CLZI(w2);
    b = 11 - z;
    w1 = (w1 >> b) | (w2 << (32-b));
    w2 >>= b;
    return as_double(((ulong)(1022 + scale - start - z) << 52) | ((ulong)(w2 & 0x000fffff) << 32) | (ulong)w1);
}

int
MATH_PRIVATE(trigredlarge)(__private double *r, __private double *rr, double x)
{
    double p0, p1, p2;

    // Scale x by relevant part of 2/pi
    if (AMD_OPT()) {
        p2 = BUILTIN_TRIG_PREOP_F64(x, 0);
        p1 = BUILTIN_TRIG_PREOP_F64(x, 1);
        p0 = BUILTIN_TRIG_PREOP_F64(x, 2);
    } else {
        const int e_clamp = 1077;
        int e = as_int2(x).y >> 20;
        int shift = e > e_clamp ?  e - e_clamp : 0;
        int scale = e >= 0x7b0 ? 128 : 0;

        p2 = get_twobypi_bits(shift,       scale);
        p1 = get_twobypi_bits(shift +  53, scale);
        p0 = get_twobypi_bits(shift + 106, scale);
    }

    if (AMD_OPT()) {
        x = BUILTIN_FLDEXP_F64(x, x >= 0x1.0p+945 ? -128 : 0);
    } else {
        x *= x >= 0x1.0p+945 ? 0x1.0p-128 : 1.0;
    }

    double f2, f1, f0, c2, c1;
    EXPAND(x, p2, p1, p0, f2, f1, f0, c2, c1);
    SHIFT(f2, f1, f0, c2, c1);

    // Remove most significant integer bits
    if (AMD_OPT()) {
        f2 = BUILTIN_FLDEXP_F64(BUILTIN_FRACTION_F64(BUILTIN_FLDEXP_F64(f2, -16)), 16);
    } else {
        f2 = BUILTIN_FRACTION_F64(f2 * 0x1.0p-16) * 0x1.0p+16;
    }

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
    const double pio2hh = 0x1.921fb50000000p+0;
    const double pio2ht = 0x1.110b460000000p-26;
    const double pio2t  = 0x1.1a62633145c07p-54;

    double rh, rt;

    if (HAVE_FAST_FMA64()) {
        rh = f2 * pio2h;
        rt = BUILTIN_FMA_F64(f1, pio2h, BUILTIN_FMA_F64(f2, pio2t, BUILTIN_FMA_F64(f2, pio2h, -rh)));
    } else { 
        double f2h = as_double(as_ulong(f2) & 0xfffffffff8000000UL);
        double f2t = f2 - f2h;

        rh = f2 * pio2h;
        rt = MATH_MAD(f2t, pio2ht, MATH_MAD(f2h, pio2ht, MATH_MAD(f2t, pio2hh, MATH_MAD(f2h, pio2hh, -rh)))) +
             MATH_MAD(f1, pio2h, f2*pio2t);
    }

    FSUM2(rh, rt, rh, rt);
    *r = rh;
    *rr = rt;

    return i & 0x3;
}

