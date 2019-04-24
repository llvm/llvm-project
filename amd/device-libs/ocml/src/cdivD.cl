/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define CP(A,B,C,D) ({ \
    double _a = A; \
    double _b = B; \
    double _c = C; \
    double _d = D; \
    double _bd = _b * _d; \
    double _e = BUILTIN_FMA_F64(_b, _d, -_bd); \
    double _f = BUILTIN_FMA_F64(_a, _c, _bd); \
    _f + _e; \
})


CONSTATTR double2
MATH_MANGLE(cdiv)(double2 zn, double2 zd)
{
    double zdx = zd.x;
    double zdy = zd.y;
    bool g = BUILTIN_ABS_F64(zdx) > BUILTIN_ABS_F64(zdy);
    int de = BUILTIN_FREXP_EXP_F64(g ? zdx : zdy);
    zdx = BUILTIN_FLDEXP_F64(zdx, -de);
    zdy = BUILTIN_FLDEXP_F64(zdy, -de);
    double u = g ? zdx : zdy;
    double v = g ? zdy : zdx;
    double d2 = BUILTIN_FMA_F64(u, u, v*v);
    double tr = CP(zn.x,  zn.y, zdx, zdy);
    double ti = CP(zn.y, -zn.x, zdx, zdy);
    double nr = BUILTIN_FREXP_MANT_F64(tr);
    double ni = BUILTIN_FREXP_MANT_F64(ti);
    int er = BUILTIN_FREXP_EXP_F64(tr);
    int ei = BUILTIN_FREXP_EXP_F64(ti);

    double rr = BUILTIN_FLDEXP_F64(MATH_DIV(nr, d2), er - de);
    double ri = BUILTIN_FLDEXP_F64(MATH_DIV(ni, d2), ei - de);

    if (!FINITE_ONLY_OPT()) {
        if (BUILTIN_ISNAN_F64(rr) && BUILTIN_ISNAN_F64(ri)) {
            if (d2 == 0.0 && (!BUILTIN_ISNAN_F64(zn.x) || !BUILTIN_ISNAN_F64(zn.y))) {
                double i = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), zd.x);
                rr = i * zn.x;
                ri = i * zn.y;
            } else if ((BUILTIN_ISINF_F64(zn.x) || BUILTIN_ISINF_F64(zn.y)) &&
                       (BUILTIN_ISFINITE_F64(zd.x) && BUILTIN_ISFINITE_F64(zd.y))) {
                double znx = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zn.x) ? 1.0 : 0.0, zn.x);
                double zny = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zn.y) ? 1.0 : 0.0, zn.y);
                rr = AS_DOUBLE(PINFBITPATT_DP64) * MATH_MAD(znx, zd.x,   zny * zd.y);
                ri = AS_DOUBLE(PINFBITPATT_DP64) * MATH_MAD(zny, zd.x,  -znx * zd.y);
            } else if ((BUILTIN_ISINF_F64(zd.x) || BUILTIN_ISINF_F64(zd.y)) &&
                       (BUILTIN_ISFINITE_F64(zn.x) && BUILTIN_ISFINITE_F64(zn.y))) {
                zdx = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zd.x) ? 1.0 : 0.0, zd.x);
                zdy = BUILTIN_COPYSIGN_F64(BUILTIN_ISINF_F64(zd.y) ? 1.0 : 0.0, zd.y);
                rr = 0.0 * MATH_MAD(zn.x, zdx,  zn.y * zdy);
                ri = 0.0 * MATH_MAD(zn.y, zdx, -zn.x * zdy);
            }
        }
    }

    return (double2)(rr, ri);
}

