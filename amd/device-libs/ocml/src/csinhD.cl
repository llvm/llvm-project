/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

#define DOUBLE_SPECIALIZATION
#include "ep.h"

extern CONSTATTR double2 MATH_PRIVATE(epexpep)(double2 z);

CONSTATTR double2
MATH_MANGLE(csinh)(double2 z)
{
    double x = BUILTIN_ABS_F64(z.x);
    double2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e42fefa39efp+0,0x1.abc9e3b39803fp-55)));
    double2 er = rcp(e);
    er = ldx(er, -4);
    double2 cx = fadd(e, er);
    double2 sx = fsub(e, er);
    double cy;
    double sy = MATH_MANGLE(sincos)(z.y, &cy);

    double cxhi = cx.hi;
    double sxhi = sx.hi;

    if (!FINITE_ONLY_OPT()) {
        bool b = x >= 0x1.6395a2079b70cp+9;
        cxhi = b ? PINF_F64 : cxhi;
        sxhi = b ? PINF_F64 : sxhi;
    }

    bool s = x >= 0x1.0p-27;
    double rr = BUILTIN_FLDEXP_F64(BUILTIN_COPYSIGN_F64(s ? sxhi : x, z.x) * cy, s);
    double ri = BUILTIN_FLDEXP_F64(cxhi * sy, 1);

    if (!FINITE_ONLY_OPT()) {
        rr = (!BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_NZER|CLASS_PINF|CLASS_NINF) |
              BUILTIN_ISFINITE_F64(z.y)) ? rr : z.x;
        ri = (BUILTIN_ISFINITE_F64(x) | (z.y != 0.0)) ? ri : z.y;
    }

    return (double2)(rr, ri);
}

