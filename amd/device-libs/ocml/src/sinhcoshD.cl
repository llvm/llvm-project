
#include "mathD.h"

PUREATTR double
MATH_PRIVATE(sinhcosh)(double y, int which)
{
    USE_TABLE(double2, sinh_tbl, M64_SINH);
    USE_TABLE(double2, cosh_tbl, M64_COSH);

    // In this range we find the integer part y0 of y
    // and the increment dy = y - y0. We then compute
    //     sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)
    //     cosh(y) = cosh(y0)cosh(dy) + sinh(y0)sinh(dy)
    // where sinh(y0) and cosh(y0) are obtained from tables

    int ind = (int)y;
    ind = ind > 36 ? 0 : ind;
    double dy = y - (double)ind;
    double dy2 = dy * dy;

    double sdy = MATH_MAD(dy2,
                     MATH_MAD(dy2,
                         MATH_MAD(dy2,
                             MATH_MAD(dy2,
                                 MATH_MAD(dy2,
                                     MATH_MAD(dy2, 0x1.b4125921ea08bp-41, 0x1.611cb2bdcb7bep-33),
                                     0x1.ae6460fbe61c0p-26),
                                 0x1.71de3a4e13e7dp-19),
                             0x1.a01a01a01ee80p-13),
                         0x1.11111111110fdp-7),
                     0x1.5555555555555p-3);
    sdy = sdy * dy * dy2;


    double cdy = MATH_MAD(dy2,
                     MATH_MAD(dy2,
                         MATH_MAD(dy2,
                             MATH_MAD(dy2,
                                 MATH_MAD(dy2,
                                     MATH_MAD(dy2, 0x1.9984b7f63fcd7p-37, 0x1.1ee56385b7b20p-29),
                                     0x1.27e5069f1cb55p-22),
                                 0x1.a01a019079011p-16),
                             0x1.6c16c16c212e5p-10),
                         0x1.5555555555502p-5),
                     0x1.0000000000000p-1);
    cdy *= dy2;

    // At this point sinh(dy) is approximated by dy + sdy.
    // and cosh(dy) is approximated by 1 + cdy.

    double2 tv = cosh_tbl[ind];
    double cl = tv.s0;
    double ct = tv.s1;
    tv = sinh_tbl[ind];
    double sl = tv.s0;
    double st = tv.s1;

    double z;
    if (which == 0) {
        double sdy1 = AS_DOUBLE(AS_ULONG(dy) & 0xfffffffff8000000UL);
        double sdy2 = sdy + (dy - sdy1);
        z = MATH_MAD(cl, sdy1, MATH_MAD(sl, cdy, MATH_MAD(cl, sdy2, MATH_MAD(ct, sdy1, MATH_MAD(st, cdy, ct*sdy2)) + st))) + sl;
    } else {
        z = MATH_MAD(sl, dy, MATH_MAD(sl, sdy, MATH_MAD(cl, cdy, MATH_MAD(st, dy, MATH_MAD(st, sdy, ct*cdy)) + ct))) + cl;
    }

    return z;
}

