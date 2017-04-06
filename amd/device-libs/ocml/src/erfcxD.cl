
#include "mathD.h"

PUREATTR double
MATH_MANGLE(erfcx)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 1.0) {
        ret = MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
                  0x1.997339112da12p-29, -0x1.9a1485b7ae337p-27),
                  0x1.9548ab4c5bb56p-26), -0x1.2f88b47e02dc3p-24),
                  0x1.282114351c39ap-22), -0x1.e533a426aadd7p-21),
                  0x1.723131b8ef11ep-19), -0x1.188f6b08d66b9p-17),
                  0x1.a00995a561233p-16), -0x1.2aeb04681fed5p-14),
                  0x1.a01b9d82bcaa5p-13), -0x1.182d3bb1ac2c8p-11),
                  0x1.6c16a932f49d1p-10), -0x1.c74aef6905182p-9),
                  0x1.111111f403407p-7), -0x1.390379458257cp-6),
                  0x1.5555554b34536p-5), -0x1.6023e8de7793p-4),
                  0x1.5555555597342p-3), -0x1.341f6bc020c17p-2),
                  0x1.fffffffffe5aep-2), -0x1.812746b037cadp-1),
                  0x1.000000000001dp0), -0x1.20dd750429b6ap0),
                  0x1.0p0);
    } else if (ax < 5120.0) {
        double t = MATH_DIV(ax - 4.0, ax + 4.0);
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t,
                  0.14981549849751462e-8, -0.69954933359042387e-8),
                  -0.15965692247743744e-7), 0.92967132363414431e-7),
                  0.70214215034531004e-7), -0.80204958740421079e-6),
                  0.29923810132862422e-6), 0.56895739871851154e-5),
                  -0.11226090578381133e-4), -0.2438781785281914e-4),
                  0.00015062360829881126), -0.00019926094025574419),
                  -0.00075777387606136804), 0.0050319709983606006),
                  -0.016197733946788412), 0.037167515387099868),
                  -0.066330365824435124), 0.093732835010698844),
                  -0.10103906603561565), 0.068097054254223675),
                  0.015379652102604634), -0.13962111684055725),
                  1.2329951186255526);
        ret = MATH_DIV(ret, MATH_MAD(ax, 2.0, 1.0));
    } else {
        const double one_over_sqrtpi = 0x1.20dd750429b6dp-1;
        double z = MATH_RCP(x * x);
        ret =  MATH_DIV(one_over_sqrtpi, x) * MATH_MAD(z, MATH_MAD(z, 0.375, -0.5), 1.0);
    }

    if (x <= -1.0) {
        double x2h = ax * ax;
        double x2l = BUILTIN_FMA_F64(ax, ax, -x2h);
        ret = MATH_MANGLE(exp)(x2h) * MATH_MANGLE(exp)(x2l) * 2.0 - ret;
        ret = x < -27.0 ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
    }

    return ret;
}

