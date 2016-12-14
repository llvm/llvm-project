/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(ncdf)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 1.0) {
        double t = x*x;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t,
                  0x1.1e87d928af1f6p-38, -0x1.e68f4029dc2c0p-34), 0x1.380a261f87551p-29), -0x1.621cb3c3f7149p-25),
                  0x1.6589f0b35f55fp-21), -0x1.3ce8f9b53d3e9p-17), 0x1.e42b0d4ae4d22p-14), -0x1.37403f6b929d4p-10),
                  0x1.46d0429768ffdp-7), -0x1.1058377e2cedfp-4), 0x1.9884533d43651p-2);
        ret = MATH_MAD(x, ret, 0.5);
    } else {
        if (ax < 2.0) {
            double t = ax - 2.0;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.3fb344e232110p-26, 0x1.34054a0bc4c6fp-24), -0x1.939cd84b381a6p-23), -0x1.b3475e4828d0dp-22),
                      0x1.d4cb2a496baf8p-19), -0x1.8222632c3dce9p-19), -0x1.3709819e7bedbp-15), 0x1.e2f4fa24e4280p-14),
                      0x1.ee3aa32b01c65p-14), -0x1.61d5de1699b15p-10), 0x1.26dcd696d1225p-9), 0x1.26dcd792e9270p-8),
                      -0x1.ba4b436eade88p-6), 0x1.ba4b436e8314cp-5), -0x1.ba4b436e83aefp-5), 0x1.74bcf82c9d860p-6);
        } else {
            double t = MATH_DIV(ax - 4.0, ax + 4.0);
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.eb166edbbcdd9p-24, -0x1.d50e59dd0a871p-22), 0x1.f30171f1e0f24p-24), 0x1.55fd1c42f2e9ep-20),
                      0x1.22d3515aa5decp-21), -0x1.8b3695f03426dp-18), -0x1.94005ca736ae9p-18), 0x1.08b397e219ae8p-15),
                      0x1.3da64be5ad668p-15), -0x1.ac8fe6827ce22p-13), -0x1.2c888b9633b2dp-13), 0x1.9d653bafa9b2cp-10),
                      -0x1.3d35c1775e54fp-10), -0x1.5877b9b55cb6cp-7), 0x1.6b6ea911dd389p-5), -0x1.862fda1beb22fp-4),
                      0x1.0800fd0f6411dp-3), -0x1.8ac5bd76365a6p-4), -0x1.78fbb7ec2d8f0p-6), 0x1.b30b52fe27788p-1);

            double x2h, x2l;
            if (HAVE_FAST_FMA64()) {
                x2h = ax * ax;
                x2l = BUILTIN_FMA_F64(ax, ax, -x2h);
            } else {
                double xh = AS_DOUBLE(AS_ULONG(ax) & 0xffffffff00000000UL);
                double xl = ax - xh;
                x2h = xh*xh;
                x2l = (ax + xh)*xl;
            }

            ret = MATH_DIV(ret, MATH_MAD(ax, 2.0, 1.0)) * MATH_MANGLE(exp)(-0.5*x2l) * MATH_MANGLE(exp)(-0.5*x2h);
            ret = BUILTIN_CLASS_F64(ax, CLASS_PINF) ? 0.0 : ret;
        }
        double retp = 1.0 - ret;
        ret = x > 0.0 ? retp : ret;
    }

    return ret;
}

