/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(erfinv)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 0.375) {
        double t = ax*ax;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  0x1.c5ec06cd8002bp-2, -0x1.bb7dd47aef0d6p-1), 0x1.d189992eccdb6p-1), -0x1.10ec180cde957p-1),
                  0x1.05cce379dd66fp-2), -0x1.6b9067e3dae74p-5), 0x1.5f7f0487c11a3p-5), 0x1.e0fbf22b2350cp-6),
                  0x1.2ce26322b7f90p-5), 0x1.5ebeeee81dd31p-5), 0x1.a7cacb897f0d4p-5), 0x1.0a130d62cba32p-4),
                  0x1.62847c8653359p-4), 0x1.053c2c0a5e083p-3), 0x1.db29fb2feec72p-3), 0x1.c5bf891b4ef6ap-1);
        ret = ax * ret;
    } else if (ax < 0x1.fffep-1) {
        double w = -MATH_MANGLE(log)(BUILTIN_FMA_F64(-ax, ax, 1.0));

        if (w < 6.25) {
            w = w - 3.125;
            ret = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w,
                      -0x1.135d2e746e627p-68, -0x1.8ddf93324d327p-63),  0x1.7b83eef0b7c9fp-60), 0x1.9ba72cd589b91p-57),
                      -0x1.33689090a6b96p-53), 0x1.82e11898132e0p-56),  0x1.de4acfd9e26bap-48), -0x1.6d33eed66c487p-45),
                      -0x1.6f2167040d8e2p-44), 0x1.72a22c2d77e20p-39), -0x1.c8859c4e5c0afp-37), -0x1.dc583d118a561p-35),
                      0x1.20f47ccf46b3cp-30), -0x1.1a9e38dc84d60p-28), -0x1.f36cd6d3d46a9p-26), 0x1.c6b4f5d03b787p-22),
                      -0x1.6e8a5434ae8a2p-20), -0x1.d1d1f7b8736f6p-17),  0x1.879c2a212f024p-13), -0x1.845769484fca8p-11),
                      -0x1.8b6c33114f909p-8), 0x1.ebd80d9b13e28p-3),   0x1.a755e7c99ae86p+0);
        } else if (w < 16.0) {
            w = MATH_SQRT(w) - 3.25;
            ret = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w,
                       0x1.3040f87dbd932p-29, 0x1.85cbe52878635p-24), -0x1.2777453dd3955p-22), 0x1.395abcd554c6cp-26),
                       0x1.936388a3790adp-20), -0x1.0d5db812b5083p-18),  0x1.8860cd5d652f6p-19), 0x1.a29a0cacdfb23p-17),
                       -0x1.8cef1f80281f2p-15), 0x1.1e684d0b9188ap-14),  0x1.932cd54c8a222p-16), -0x1.7448a89ef8aa3p-12),
                       0x1.f3cc55ad40c25p-11), -0x1.ba924132f38b1p-10),  0x1.468eeca533cf8p-9), -0x1.ebadabb891bbdp-9),
                       0x1.5ffcfe5b76afcp-8), 0x1.0158a6d641d39p+0),   0x1.8abcc380d5a48p+1);
        } else {
            w = MATH_SQRT(w) - 5.0;
            ret = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                  MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                      -0x1.dcec3a7785389p-36, -0x1.18feec0e38727p-32),  0x1.9e6bf2dda45e3p-30), -0x1.0468fb24e2f5fp-28),
                      0x1.05ac6a8fba182p-27), -0x1.0102e495fb9c0p-26),  0x1.f4c20e1334af8p-26), -0x1.22d220fdf9c3ep-24),
                      0x1.ebc8bb824cb54p-23), -0x1.0a8d40ea372ccp-20),  0x1.2fbd29d093d2bp-18), -0x1.4a3497e1e0facp-16),
                      0x1.3ebf4eb00938fp-14), -0x1.c2f36a8fc5d53p-13), -0x1.22ea5df04047cp-13), 0x1.02a30d1fba0dcp+0),
                      0x1.3664ddd1ad7fbp+2);
        }
        ret = ax * ret;
    } else {
        double s = MATH_SQRT(-MATH_MANGLE(log)(1.0 - ax));
        double t = MATH_RCP(s);

        if (ax < 0x1.fffffffep-1) {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.c4bd831a51669p+7, -0x1.66af45b757c26p+9), 0x1.061b293ee1671p+10), -0x1.d4aa0fd7248e9p+9),
                      0x1.1eebb0088748dp+9), -0x1.ff4cb6c165efep+7), 0x1.59c379a609255p+6), -0x1.762b2677680c6p+4),
                      0x1.7626132cf7c5ap+2), -0x1.a298cc231a949p+0), -0x1.9fa2d429b22cap-6), 0x1.00131c4b15d15p+0);
        } else {
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, 
                      0x1.e1f462cc8e58ap+7, -0x1.dd260d25bee8dp+8), 0x1.af7dab6c206e6p+8), -0x1.d97c75a0f5809p+7),
                      0x1.632c20bf45d30p+6), -0x1.8e4908179a727p+4), 0x1.89538a73a2c3cp+2), -0x1.aad8569b3607dp+0),
                      -0x1.80d1bec4b54cbp-6), 0x1.001006f90ea2cp+0);
        }

        ret = s * ret;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = ax > 1.0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax == 1.0 ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
    }

    return BUILTIN_COPYSIGN_F64(ret, x);
}

