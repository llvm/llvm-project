/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(erf)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 1.0) {
        double t = ax * ax;
        double p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                   MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                   MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                       -0x1.ab15c51d2ebebp-31, 0x1.d6e3ddfeb1f49p-27),
                       -0x1.5bfe76384472p-23), 0x1.b97e44280cfb9p-20),
                       -0x1.f4ca204c771c5p-17), 0x1.f9a2b75531772p-14),
                       -0x1.c02db0149d904p-11), 0x1.565bccf7e2856p-8),
                       -0x1.b82ce311ee09bp-6), 0x1.ce2f21a0408d1p-4),
                       -0x1.812746b0379b2p-2), 0x1.06eba8214db68p-3);
        ret = MATH_MAD(ax, p, ax);
    } else {
        double p = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
                   MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
                   MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
                   MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
                   MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
                   MATH_MAD(ax, MATH_MAD(ax,
                        0x1.98d37c14b24bep-58, -0x1.145a3502a41cdp-51),
                        0x1.62deed735f9ecp-46), -0x1.1ffe55552ca22p-41),
                        0x1.4b9ba7074b644p-37), -0x1.20345a78ce24p-33),
                        0x1.88b7a0cefddd8p-30), -0x1.aded48c94b617p-27),
                        0x1.803aa312306dp-24), -0x1.1b0106f4c5a9bp-21),
                        0x1.58c0e7cfd79aep-19), -0x1.59e386410fdf7p-17),
                        0x1.192fc1f9b1786p-15), -0x1.62cf3f4634b2ep-14),
                        0x1.314dfb42f7e4bp-13), -0x1.2cb68c047288ap-14),
                        -0x1.038ff7bbcce25p-11), 0x1.a9466ae1babaep-10),
                        -0x1.58be1e65a6063p-13), -0x1.39bc16738ee3ap-6),
                        0x1.a4fbc28146b69p-4), 0x1.45f2da69750c4p-1),
                        0x1.06ebb919fcca8p-3);
        p = MATH_MAD(ax, p, ax);
        ret = 1.0 - MATH_MANGLE(exp)(-p);
    }

    ret = BUILTIN_COPYSIGN_F64(ret, x);
    return ret;
}

