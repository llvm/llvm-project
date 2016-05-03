
#include "mathD.h"
#define REDUCE_DIVISION

#define FULL_MUL(A, B, CHI, CLO) \
    do { \
        if (HAVE_FAST_FMA32()) { \
            CHI = A * B; \
            CLO = BUILTIN_FMA_F64(A, B, -CHI); \
        } else { \
            double __ha = as_double(as_ulong(A) & 0xfffffffff8000000UL); \
            double __ta = A - __ha; \
            double __hb = as_double(as_ulong(B) & 0xfffffffff8000000UL); \
            double __tb = B - __hb; \
            CHI = A * B; \
            CLO = MATH_MAD(__ta, __tb, MATH_MAD(__ta, __hb, MATH_MAD(__ha, __tb, MATH_MAD(__ha, __hb, -CHI)))); \
        } \
    } while (0)

PUREATTR double
MATH_MANGLE(tanh)(double x)
{
    // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
    // to the following three formulae:
    // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
    // 2.  (1 - (2/(exp(2*x) + 1 )))
    // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
    // but computationally, some formulae are better on some ranges.

    double y = BUILTIN_ABS_F64(x);
    double y2 = x * x;
    double z;

#if !defined REDUCE_DIVISION
    if (y <= 1.0) {
        double zn, zd;
        if (y < 0.9) {
            zn = MATH_MAD(y2,
                     MATH_MAD(y2,
                         MATH_MAD(y2, -0x1.e82d10d09af81p-27, -0x1.a387bfaf479c9p-13),
                         -0x1.20629b90302aep-6),
                     -0x1.189b6e8007758p-2);

            zd = MATH_MAD(y2,
                     MATH_MAD(y2,
                         MATH_MAD(y2, 0x1.b68b3cecad284p-13, 0x1.4a3d4cc7a88a9p-6),
                         0x1.86cd01c4ab94cp-2),
                     0x1.a4e925c00b304p-1);
        } else {
            zn = MATH_MAD(y2,
                     MATH_MAD(y2,
                         MATH_MAD(y2, -0x1.8cc5a847e6cecp-27, -0x1.5b483c69835ddp-13),
                         -0x1.defad6e212118p-7),
                     -0x1.d28597c5ae288p-3);

            zd = MATH_MAD(y2,
                     MATH_MAD(y2,
                         MATH_MAD(y2, 0x1.6af77f334e750p-13, 0x1.12335baec487fp-6),
                         0x1.44d1459bcc5f6p-2),
                     0x1.5de431d442afdp-1);
        }

        z = MATH_MAD(y*y2, MATH_FAST_DIV(zn, zd), y);
    } else {
        double p = MATH_MANGLE(exp)(2.0 * y) + 1.0;
        z = 1.0 - MATH_FAST_DIV(2.0, p);
        z = y > 0x1.2b708872320e2p+4 ? 1.0 : z;
    }

    // Other cases
    if (!FINITE_ONLY_OPT()) {
        z = (y < 0x1.0p-28) | BUILTIN_CLASS_F64(y, CLASS_SNAN|CLASS_QNAN) ? x : z;
    } else {
        z = y < 0x1.0p-28 ? x : z;
    }
#else
    if (y < 0.3) {
        double p;
        p = MATH_MAD(y2,
                MATH_MAD(y2,
                    MATH_MAD(y2,
                        MATH_MAD(y2,
                            MATH_MAD(y2,
                                MATH_MAD(y2,
                                    MATH_MAD(y2,
                                        MATH_MAD(y2,
                                            MATH_MAD(y2,
                                                MATH_MAD(y2, 0x1.967e18afcafadp-14, -0x1.f57d7734d1664p-13),
                                                0x1.3558248036744p-11),
                                            -0x1.7da36452b75e3p-10),
                                        0x1.d6d3d0e157de0p-9),
                                    -0x1.226e355e6c23dp-7),
                                0x1.664f4882c10fap-6),
                            -0x1.ba1ba1ba1ba1cp-5),
                        0x1.1111111111111p-3),
                    -0x1.5555555555555p-2),
                1.0);
        z = y * p;
    } else if (y < 0.9) {
        double p;
        p = MATH_MAD(y2,
                MATH_MAD(y2,
                    MATH_MAD(y2,
                        MATH_MAD(y2,
                            MATH_MAD(y2,
                                MATH_MAD(y2,
                                    MATH_MAD(y2,
                                        MATH_MAD(y2,
                                            MATH_MAD(y2,
                                                MATH_MAD(y2,
                                                    MATH_MAD(y2,
                                                        MATH_MAD(y2,
                                                            MATH_MAD(y2, 0x1.1bd6cc34f0107p-22, -0x1.370dfd7dc8e69p-19),
                                                            0x1.5f48f4d8b2027p-17),
                                                        -0x1.1ccced3ffb092p-15),
                                                    0x1.863a8ef274bb6p-14),
                                                -0x1.f10faf9ea7b11p-13),
                                            0x1.34e4818ccddb2p-11),
                                        -0x1.7d91cdb46964fp-10),
                                    0x1.d6d1d82ae9267p-9),
                                -0x1.226e20f4f9d8cp-7),
                            0x1.664f475c48a83p-6),
                        -0x1.ba1ba1af376c5p-5),
                    0x1.11111110f2d9ep-3),
                -0x1.55555555550b5p-2);

        double y2h, y2t, y3h, y3t;
        FULL_MUL(y, y, y2h, y2t);
        FULL_MUL(y, y2h, y3h, y3t);
        y3t += y * y2t;
        double t = y3h;
        y3h = y3h + y3t;
        y3t = y3t - (y3h - t);
        z = MATH_MAD(y3t, p,MATH_MAD(y3h, p, y));
    } else if (y < 1.0) {
        double p;
        p = MATH_MAD(y2,
                MATH_MAD(y2,
                    MATH_MAD(y2,
                        MATH_MAD(y2,
                            MATH_MAD(y2,
                                MATH_MAD(y2,
                                    MATH_MAD(y2,
                                        MATH_MAD(y2, -0x1.e2e8cc873a840p-17, 0x1.404d13968404ap-13),
                                        -0x1.a3e686f98a528p-11),
                                    0x1.7838b92a630e6p-9),
                                -0x1.0fc4658f256a9p-7),
                            0x1.613261681d5a7p-6),
                        -0x1.b92f593ce27dcp-5),
                    0x1.11046d76b7f1cp-3),
                -0x1.5554b9dadba3fp-2);

        z = MATH_MAD(p, y2*y, y);
    } else if (y < 19.1) {
        double t = MATH_MANGLE(exp)(y+y) + 1.0;
        z = 1.0 - MATH_FAST_DIV(2.0, t);
    } else {
        z = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN) ? x : 1.0;
    }
#endif
    return BUILTIN_COPYSIGN_F64(z, x);
}

