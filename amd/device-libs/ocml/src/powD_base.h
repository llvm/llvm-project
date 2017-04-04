/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double2 MATH_PRIVATE(epln)(double);
extern CONSTATTR double MATH_PRIVATE(expep)(double2);

#define DOUBLE_SPECIALIZATION
#include "ep.h"

PUREATTR double
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(double x, double y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(double x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(double x, int ny)
#else
MATH_MANGLE(pow)(double x, double y)
#endif
{
#if defined(COMPILING_POWN)
    double y = (double) ny;
#elif defined(COMPILING_ROOTN)
    double2 y = rcp((double)ny);
#endif

    double ax = BUILTIN_ABS_F64(x);
    double expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

    // y status: 0=not integer, 1=odd, 2=even
#if defined(COMPILING_POWN) | defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    double ay = BUILTIN_ABS_F64(y);
    int inty;
    {
        double tay = BUILTIN_TRUNC_F64(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F64(tay*0.5) == 0.0);
    }
#endif

    double ret = BUILTIN_COPYSIGN_F64(expylnx, (inty == 1) & (x < 0.0) ? -0.0 : 0.0);

    // Now all the edge cases
#if defined COMPILING_POWR
    bool ax_eq_0 = ax == 0.0;
    bool ax_ne_0 = ax != 0.0;
    bool ax_lt_1 = ax < 1.0;
    bool ax_eq_1 = ax == 1.0;
    bool ax_gt_1 = ax > 1.0;
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool ay_eq_0 = ay == 0.0;
    bool ay_eq_pinf = BUILTIN_CLASS_F64(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F64(ay, CLASS_QNAN|CLASS_SNAN);
    bool y_eq_ninf = BUILTIN_CLASS_F64(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F64(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F64(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F64(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = ax_lt_1 & y_eq_ninf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0 : ret;
        ret = ax_eq_1 & ay_lt_inf ? 1.0 : ret;
        ret = ax_eq_1 & ay_eq_pinf ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0 : ret;
        ret = ax_gt_1 & y_eq_pinf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_lt_pinf & ay_eq_0 ? 1.0 : ret;
        ret = ax_eq_pinf & !y_pos ? 0.0 : ret;
        ret = ax_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_pinf & y_eq_pinf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_pinf & ay_eq_0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_eq_0 & !y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_0 & y_pos ? 0.0 : ret;
        ret = ax_eq_0 & ay_eq_0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_ne_0 & !x_pos ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
	ret = ax_eq_1 ? 1.0 : ret;
	ret = ay_eq_0 ? 1.0 : ret;
	ret = ax_eq_0 & y_pos ? 0.0 : ret;
    }
#elif defined COMPILING_POWN
    bool ax_eq_0 = ax == 0.0;
    bool x_eq_ninf = BUILTIN_CLASS_F64(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        double xinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0 : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0 : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0 : ret;
        ret = x_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
    } else {
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
    }
    ret = ny == 0 ? 1.0 : ret;
#elif defined COMPILING_ROOTN
    bool ax_eq_0 = ax == 0.0;
    bool x_eq_ninf = BUILTIN_CLASS_F64(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 2) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        double xinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0 : ret;
        ret = x_eq_pinf & !y_pos ? 0.0 : ret;
        ret = x_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ny == 0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
    } else {
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0 : ret;
    }
#else
    bool ax_eq_0 = ax == 0.0;
    bool ax_ne_0 = ax != 0.0;
    bool ax_lt_1 = ax < 1.0;
    bool ax_eq_1 = ax == 1.0;
    bool ax_gt_1 = ax > 1.0;
    bool ax_lt_pinf = BUILTIN_CLASS_F64(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F64(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F64(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool x_eq_ninf = BUILTIN_CLASS_F64(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F64(x, CLASS_PINF);
    bool ay_eq_0 = ay == 0.0;
    bool ay_eq_pinf = BUILTIN_CLASS_F64(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F64(ay, CLASS_QNAN|CLASS_SNAN);
    bool y_eq_ninf = BUILTIN_CLASS_F64(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F64(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F64(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F64(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 0) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = ax_lt_1 & y_eq_ninf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0 : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0 : ret;
        ret = ax_gt_1 & y_eq_pinf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        double xinf = BUILTIN_COPYSIGN_F64(AS_DOUBLE(PINFBITPATT_DP64), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty != 1) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0 : ret;
        ret = ax_eq_0 & y_eq_ninf ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = (x == -1.0) & ay_eq_pinf ? 1.0 : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0 : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0 : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0 : ret;
        ret = x_eq_pinf & y_pos ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
        double xzero = BUILTIN_COPYSIGN_F64(0.0, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0 : ret;
    }
    ret = ay_eq_0 ? 1.0 : ret;
    ret = x == 1.0 ? 1.0 : ret;
#endif

    return ret;
}

