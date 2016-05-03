
#include "mathH.h"

PUREATTR half
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(half x, half y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(half x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(half x, int ny)
#else
MATH_MANGLE(pow)(half x, half y)
#endif
{
    half ax = BUILTIN_ABS_F16(x);

#if defined(COMPILING_POWN)
    float fy = (float)ny;
#elif defined(COMPILING_ROOTN)
    float fy;
    if (AMD_OPT()) {
        fy = BUILTIN_RCP_F32((float)ny);
    } else {
        fy = BUILTIN_DIV_F32(1.0f, (float)ny);
    }
#else
    float fy = (float)y;
#endif

    float p;
    if (AMD_OPT()) {
        p = BUILTIN_EXP2_F32(fy * BUILTIN_LOG2_F32((float)ax));
    } else {
        p = MATH_UPMANGLE(exp2)(fy * MATH_UPMANGLE(log2)((float)ax));
    }

    // Classify y:
    //   inty = 0 means not an integer.
    //   inty = 1 means odd integer.
    //   inty = 2 means even integer.

#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    half ay = BUILTIN_ABS_F16(y);
    int inty;
    {
        half tay = BUILTIN_TRUNC_F16(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F16(tay*0.5h) == 0.0h);
    }
#endif

    half ret = BUILTIN_COPYSIGN_F16((half)p, (inty == 1) & (x < 0.0h) ? -0.0f : 0.0f);

    // Now all the edge cases
#if defined COMPILING_POWR
    bool ax_eq_0 = ax == 0.0h;
    bool ax_ne_0 = ax != 0.0h;
    bool ax_lt_1 = ax < 1.0h;
    bool ax_eq_1 = ax == 1.0h;
    bool ax_gt_1 = ax > 1.0h;
    bool ax_lt_pinf = BUILTIN_CLASS_F16(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F16(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool ay_eq_0 = ay == 0.0h;
    bool ay_eq_pinf = BUILTIN_CLASS_F16(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F16(ay, CLASS_QNAN|CLASS_SNAN);
    bool y_eq_ninf = BUILTIN_CLASS_F16(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F16(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F16(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F16(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = ax_lt_1 & y_eq_ninf ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0h : ret;
        ret = ax_eq_1 & ay_lt_inf ? 1.0h : ret;
        ret = ax_eq_1 & ay_eq_pinf ? as_half((ushort)QNANBITPATT_HP16) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0h : ret;
        ret = ax_gt_1 & y_eq_pinf ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_lt_pinf & ay_eq_0 ? 1.0h : ret;
        ret = ax_eq_pinf & !y_pos ? 0.0h : ret;
        ret = ax_eq_pinf & y_pos ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_pinf & y_eq_pinf ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_pinf & ay_eq_0 ? as_half((ushort)QNANBITPATT_HP16) : ret;
        ret = ax_eq_0 & !y_pos ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_0 & y_pos ? 0.0h : ret;
        ret = ax_eq_0 & ay_eq_0 ? as_half((ushort)QNANBITPATT_HP16) : ret;
        ret = ax_ne_0 & !x_pos ? as_half((ushort)QNANBITPATT_HP16) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
	ret = ax_eq_1 ? 1.0h : ret;
	ret = ay_eq_0 ? 1.0h : ret;
	ret = ax_eq_0 & y_pos ? 0.0h : ret;
    }
#elif defined COMPILING_POWN
    bool ax_eq_0 = ax == 0.0h;
    bool x_eq_ninf = BUILTIN_CLASS_F16(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F16(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F16(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        half xinf = BUILTIN_COPYSIGN_F16(as_half((ushort)PINFBITPATT_HP16), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0h : ret;
        half xzero = BUILTIN_COPYSIGN_F16(0.0h, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0h : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0h : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? as_half((ushort)NINFBITPATT_HP16) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0h : ret;
        ret = x_eq_pinf & y_pos ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_nan ? x : ret;
    } else {
        half xzero = BUILTIN_COPYSIGN_F16(0.0h, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0h : ret;
    }
    ret = ny == 0 ? 1.0h : ret;
#elif defined COMPILING_ROOTN
    bool ax_eq_0 = ax == 0.0h;
    bool x_eq_ninf = BUILTIN_CLASS_F16(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F16(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F16(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 2) ? as_half((ushort)QNANBITPATT_HP16) : ret;
        half xinf = BUILTIN_COPYSIGN_F16(as_half((ushort)PINFBITPATT_HP16), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0h : ret;
        half xzero = BUILTIN_COPYSIGN_F16(0.0h, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? as_half((ushort)NINFBITPATT_HP16) : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0h : ret;
        ret = x_eq_pinf & !y_pos ? 0.0h : ret;
        ret = x_eq_pinf & y_pos ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ny == 0 ? as_half((ushort)QNANBITPATT_HP16) : ret;
    } else {
        half xzero = BUILTIN_COPYSIGN_F16(0.0h, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0h : ret;
    }
#else
    bool ax_eq_0 = ax == 0.0h;
    bool ax_ne_0 = ax != 0.0h;
    bool ax_lt_1 = ax < 1.0h;
    bool ax_eq_1 = ax == 1.0h;
    bool ax_gt_1 = ax > 1.0h;
    bool ax_lt_pinf = BUILTIN_CLASS_F16(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F16(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool x_eq_ninf = BUILTIN_CLASS_F16(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F16(x, CLASS_PINF);
    bool ay_eq_0 = ay == 0.0h;
    bool ay_eq_pinf = BUILTIN_CLASS_F16(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F16(ay, CLASS_QNAN|CLASS_SNAN);
    bool y_eq_ninf = BUILTIN_CLASS_F16(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F16(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F16(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F16(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 0) ? as_half((ushort)QNANBITPATT_HP16) : ret;
        ret = ax_lt_1 & y_eq_ninf ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0h : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0h : ret;
        ret = ax_gt_1 & y_eq_pinf ? as_half((ushort)PINFBITPATT_HP16) : ret;
        half xinf = BUILTIN_COPYSIGN_F16(as_half((ushort)PINFBITPATT_HP16), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty != 1) ? as_half((ushort)PINFBITPATT_HP16) : ret;
        half xzero = BUILTIN_COPYSIGN_F16(0.0h, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0h : ret;
        ret = ax_eq_0 & y_eq_ninf ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = (x == -1.0h) & ay_eq_pinf ? 1.0h : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0h : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0h : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? as_half((ushort)NINFBITPATT_HP16) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0h : ret;
        ret = x_eq_pinf & y_pos ? as_half((ushort)PINFBITPATT_HP16) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
        // XXX work around conformance test incorrectly checking these cases
        half xinf = BUILTIN_COPYSIGN_F16(as_half((ushort)PINFBITPATT_HP16), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty != 1) ? as_half((ushort)PINFBITPATT_HP16) : ret;

        half xzero = BUILTIN_COPYSIGN_F16(0.0h, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0h : ret;
    }
    ret = ay == 0.0h ? 1.0h : ret;
    ret = x == 1.0h ? 1.0h : ret;
#endif

    return ret;
}

