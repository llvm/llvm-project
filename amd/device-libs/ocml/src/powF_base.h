/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epln)(float);
extern CONSTATTR float MATH_PRIVATE(expep)(float2);

CONSTATTR float
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(float x, float y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(float x, int ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(float x, int ny)
#else
MATH_MANGLE(pow)(float x, float y)
#endif
{
    float ax = BUILTIN_ABS_F32(x);
    float expylnx;

    if (UNSAFE_MATH_OPT()) {
#if defined COMPILING_POWN
        float y = (float)ny;
#elif defined COMPILING_ROOTN
        float y = MATH_FAST_RCP((float)ny);
#endif
        if (DAZ_OPT()) {
            expylnx = BUILTIN_EXP2_F32(y * BUILTIN_LOG2_F32(ax));
        } else {
            bool b = ax < 0x1.0p-126f;
            float ylnx = y * (BUILTIN_LOG2_F32(ax * (b ? 0x1.0p+24f : 1.0f)) - (b ? 24.0f : 0.0f));
            b = ylnx < -126.0f;
            expylnx = BUILTIN_EXP2_F32(ylnx + (b ? 24.0f : 0.0f)) * (b ? 0x1.0p-24f : 1.0f);
        }
    } else {
#if defined COMPILING_POWN || defined COMPILING_ROOTN
        int nyh = ny & 0xffff0000;
        float2 y = fadd((float)nyh, (float)(ny - nyh));
#if defined(COMPILING_ROOTN)
        y = rcp(y);
#endif
#endif

        expylnx = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));
    }

    // y status: 0=not integer, 1=odd, 2=even
#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    int inty = 2 - (ny & 1);
#else
    float ay = BUILTIN_ABS_F32(y);
    int inty;
    {
        float tay = BUILTIN_TRUNC_F32(ay);
        inty = ay == tay;
        inty += inty & (BUILTIN_FRACTION_F32(tay*0.5f) == 0.0f);
    }
#endif

    float ret = BUILTIN_COPYSIGN_F32(expylnx, (inty == 1) & (x < 0.0f) ? -0.0f : 0.0f);

    // Now all the edge cases
#if defined COMPILING_POWR
    bool ax_eq_0 = ax == 0.0f;
    bool ax_ne_0 = ax != 0.0f;
    bool ax_lt_1 = ax < 1.0f;
    bool ax_eq_1 = ax == 1.0f;
    bool ax_gt_1 = ax > 1.0f;
    bool ax_lt_pinf = BUILTIN_CLASS_F32(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_ISNAN_F32(x);
    bool x_pos = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool ay_eq_0 = ay == 0.0f;
    bool ay_eq_pinf = BUILTIN_CLASS_F32(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_ISNAN_F32(ay);
    bool y_eq_ninf = BUILTIN_CLASS_F32(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F32(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F32(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F32(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = ax_lt_1 & y_eq_ninf ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0f : ret;
        ret = ax_eq_1 & ay_lt_inf ? 1.0f : ret;
        ret = ax_eq_1 & ay_eq_pinf ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0f : ret;
        ret = ax_gt_1 & y_eq_pinf ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_lt_pinf & ay_eq_0 ? 1.0f : ret;
        ret = ax_eq_pinf & !y_pos ? 0.0f : ret;
        ret = ax_eq_pinf & y_pos ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_pinf & y_eq_pinf ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_pinf & ay_eq_0 ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = ax_eq_0 & !y_pos ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_0 & y_pos ? 0.0f : ret;
        ret = ax_eq_0 & ay_eq_0 ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = ax_ne_0 & !x_pos ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
	ret = ax_eq_1 ? 1.0f : ret;
	ret = ay_eq_0 ? 1.0f : ret;
	ret = ax_eq_0 & y_pos ? 0.0f : ret;
    }
#elif defined COMPILING_POWN
    bool ax_eq_0 = ax == 0.0f;
    bool x_eq_ninf = BUILTIN_CLASS_F32(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F32(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_ISNAN_F32(x);
    bool x_pos = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        float xinf = BUILTIN_COPYSIGN_F32(AS_FLOAT(PINFBITPATT_SP32), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0f : ret;
        float xzero = BUILTIN_COPYSIGN_F32(0.0f, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0f : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0f : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_FLOAT(NINFBITPATT_SP32) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0f : ret;
        ret = x_eq_pinf & y_pos ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_nan ? x : ret;
    } else {
        float xzero = BUILTIN_COPYSIGN_F32(0.0f, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0f : ret;
    }
    ret = ny == 0 ? 1.0f : ret;
#elif defined COMPILING_ROOTN
    bool ax_eq_0 = ax == 0.0f;
    bool x_eq_ninf = BUILTIN_CLASS_F32(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ax_lt_pinf = BUILTIN_CLASS_F32(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_ISNAN_F32(x);
    bool x_pos = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool y_pos = ny >= 0;

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 2) ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        float xinf = BUILTIN_COPYSIGN_F32(AS_FLOAT(PINFBITPATT_SP32), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty == 2) ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0f : ret;
        float xzero = BUILTIN_COPYSIGN_F32(0.0f, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_FLOAT(NINFBITPATT_SP32) : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0f : ret;
        ret = x_eq_pinf & !y_pos ? 0.0f : ret;
        ret = x_eq_pinf & y_pos ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ny == 0 ? AS_FLOAT(QNANBITPATT_SP32) : ret;
    } else {
        float xzero = BUILTIN_COPYSIGN_F32(0.0f, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty == 2) ? 0.0f : ret;
    }
#else
    bool ax_eq_0 = ax == 0.0f;
    bool ax_ne_0 = ax != 0.0f;
    bool ax_lt_1 = ax < 1.0f;
    bool ax_eq_1 = ax == 1.0f;
    bool ax_gt_1 = ax > 1.0f;
    bool ax_lt_pinf = BUILTIN_CLASS_F32(x, CLASS_PNOR|CLASS_PSUB);
    bool ax_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ax_eq_nan = BUILTIN_ISNAN_F32(x);
    bool x_pos = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool x_eq_ninf = BUILTIN_CLASS_F32(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ay_eq_0 = ay == 0.0f;
    bool ay_eq_pinf = BUILTIN_CLASS_F32(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_ISNAN_F32(ay);
    bool y_eq_ninf = BUILTIN_CLASS_F32(y, CLASS_NINF);
    bool y_eq_pinf = BUILTIN_CLASS_F32(y, CLASS_PINF);
    bool ay_lt_inf = BUILTIN_CLASS_F32(y, CLASS_PNOR|CLASS_PSUB);
    bool y_pos = BUILTIN_CLASS_F32(y, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);

    if (!FINITE_ONLY_OPT()) {
        ret = !x_pos & (inty == 0) ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = ax_lt_1 & y_eq_ninf ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_gt_1 & y_eq_ninf ? 0.0f : ret;
        ret = ax_lt_1 & y_eq_pinf ? 0.0f : ret;
        ret = ax_gt_1 & y_eq_pinf ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        float xinf = BUILTIN_COPYSIGN_F32(AS_FLOAT(PINFBITPATT_SP32), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty != 1) ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        float xzero = BUILTIN_COPYSIGN_F32(0.0f, x);
        ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
        ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0f : ret;
        ret = ax_eq_0 & y_eq_ninf ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = (x == -1.0f) & ay_eq_pinf ? 1.0f : ret;
        ret = x_eq_ninf & !y_pos & (inty == 1) ? -0.0f : ret;
        ret = x_eq_ninf & !y_pos & (inty != 1) ? 0.0f : ret;
        ret = x_eq_ninf & y_pos & (inty == 1) ? AS_FLOAT(NINFBITPATT_SP32) : ret;
        ret = x_eq_ninf & y_pos & (inty != 1) ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = x_eq_pinf & !y_pos ? 0.0f : ret;
        ret = x_eq_pinf & y_pos ? AS_FLOAT(PINFBITPATT_SP32) : ret;
        ret = ax_eq_nan ? x : ret;
        ret = ay_eq_nan ? y : ret;
    } else {
        // XXX work around conformance test incorrectly checking these cases
        float xinf = BUILTIN_COPYSIGN_F32(AS_FLOAT(PINFBITPATT_SP32), x);
        ret = ax_eq_0 & !y_pos & (inty == 1) ? xinf : ret;
        ret = ax_eq_0 & !y_pos & (inty != 1) ? AS_FLOAT(PINFBITPATT_SP32) : ret;

        float xzero = BUILTIN_COPYSIGN_F32(0.0f, x);
	ret = ax_eq_0 & y_pos & (inty == 1) ? xzero : ret;
	ret = ax_eq_0 & y_pos & (inty != 1) ? 0.0f : ret;
    }
    ret = ay == 0.0f ? 1.0f : ret;
    ret = x == 1.0f ? 1.0f : ret;
#endif

    return ret;
}

