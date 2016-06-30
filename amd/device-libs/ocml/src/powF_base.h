
#include "mathF.h"

// compute pow using log and exp
// x^y = exp(y * log(x))
//
// we take care not to lose precision in the intermediate steps
//
// When computing log, calculate it in splits,
//
// r = f * (p_invead + p_inv_tail)
// r = rh + rt
//
// calculate log polynomial using r, in end addition, do
// poly = poly + ((rh-r) + rt)
//
// lth = -r
// ltt = ((xexp * log2_t) - poly) + logT
// lt = lth + ltt
//
// lh = (xexp * log2_h) + logH
// l = lh + lt
//
// Calculate final log answer as gh and gt,
// gh = l & higher-half bits
// gt = (((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh))
//
// yh = y & higher-half bits
// yt = y - yh
//
// Before entering computation of exp,
// vs = ((yt*gt + yt*gh) + yh*gt)
// v = vs + yh*gh
// vt = ((yh*gh - v) + vs)
//
// In calculation of exp, add vt to r that is used for poly
// At the end of exp, do
// ((((expT * poly) + expT) + expH*poly) + expH)

PUREATTR float
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
    USE_TABLE(float2, p_log, M32_LOGE);
    USE_TABLE(float2, p_inv, M32_LOG_INV_EP);
    USE_TABLE(float2, p_jby64, M32_EXP_EP);

    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
    }

#if defined(COMPILING_POWN)
    float y = (float)ny;
#elif defined(COMPILING_ROOTN)
    float y = MATH_FAST_RCP((float)ny);
#endif

    float ax = BUILTIN_ABS_F32(x);

    // Extra precise log calculation
    float r = 1.0f - ax;
    float lth, ltt, lt, lh, l, poly;
    float2 tv;
    int m;

    if (MATH_MANGLE(fabs)(r) > 0x1.0p-4f) {
        int ixn;
        float mfn;
        
        if (AMD_OPT()) {
            mfn = (float)(BUILTIN_FREXP_EXP_F32(ax) - 1);
            ixn = AS_INT(BUILTIN_FREXP_MANT_F32(ax)) + (1 << EXPSHIFTBITS_SP32);
        } else {
            if (DAZ_OPT()) {
                mfn = (float)((AS_INT(ax) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32);
                ixn = AS_INT(ax);
            } else {
                m = (AS_INT(ax) >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
                float mf = (float)m;
                int ixs = AS_INT(AS_FLOAT(AS_INT(ax) | ONEEXPBITS_SP32) - 1.0f);
                float mfs = (float)((ixs >> EXPSHIFTBITS_SP32) - 253);
                bool c = m == -127;
                ixn = c ? ixs : ax;
                mfn = c ? mfs : mf;
            }
        }

        int indx = (ixn & 0x007f0000) + ((ixn & 0x00008000) << 1);

        // F - Y
        float f = AS_FLOAT(HALFEXPBITS_SP32 | indx) - AS_FLOAT(HALFEXPBITS_SP32 | (ixn & MANTBITS_SP32));

        indx >>= 16;
        tv = p_inv[indx];
        float rh = f * tv.s0;
        float rt = f * tv.s1;
        r = rh + rt;

        poly = MATH_MAD(r, MATH_MAD(r, 0x1.0p-2f, 0x1.555556p-2f), 0x1.0p-1f) * (r*r);
        poly += (rh - r) + rt;

        const float LOG2_HEAD = 0x1.62e000p-1f;  // 0.693115234
        const float LOG2_TAIL = 0x1.0bfbe8p-15f; // 0.0000319461833
        tv = p_log[indx];
        lth = -r;
        ltt = MATH_MAD(mfn, LOG2_TAIL, -poly) + tv.s1;
        lt = lth + ltt;
        lh = MATH_MAD(mfn, LOG2_HEAD, tv.s0);
        l = lh + lt;
    } else {
        float r2 = r*r;

        poly = MATH_MAD(r,
                   MATH_MAD(r,
                       MATH_MAD(r,
                           MATH_MAD(r, 0x1.24924ap-3f, 0x1.555556p-3f),
                           0x1.99999ap-3f),
                       0x1.000000p-2f),
                   0x1.555556p-2f);

        poly *= r2*r;

        lth = -r2 * 0.5f;
        ltt = -poly;
        lt = lth + ltt;
        lh = -r;
        l = lh + lt;
    }
    
    // Final split of log
    float gh = AS_FLOAT(AS_INT(l) & 0xfffff000);
    float gt = ((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh);

    // Now split y
    float yh = AS_FLOAT(AS_INT(y) & 0xfffff000);

#if defined(COMPILING_POWN)
    float yt = (float)(ny - (int)yh);
#elif defined(COMPILING_ROOTN)
    float fny = (float)ny;
    float fnyh = AS_FLOAT(AS_INT(fny) & 0xfffff000);
    float fnyt = (float)(ny - (int)fnyh);
    float yt = MATH_FAST_DIV(MATH_MAD(-fnyt, yh, MATH_MAD(-fnyh, yh, 1.0f)), fny);
#else
    float yt = y - yh;
#endif

    // Compute high precision y*log(x)
    float ylogx_s = MATH_MAD(gt, yh, MATH_MAD(gh, yt, yt*gt));
    float ylogx = MATH_MAD(yh, gh, ylogx_s);
    float ylogx_t = MATH_MAD(yh, gh, -ylogx) + ylogx_s;

    // Now exponentiate
    const float R_64_BY_LOG2 = 0x1.715476p+6f; // 64/log2 : 92.332482616893657
    int n = (int)(ylogx * R_64_BY_LOG2);
    float nf = (float) n;

    int j = n & 0x3f;
    m = n >> 6;

    const float R_LOG2_BY_64_LD = 0x1.620000p-7f;  // log2/64 lead: 0.0108032227
    const float R_LOG2_BY_64_TL = 0x1.c85fdep-16f; // log2/64 tail: 0.0000272020388
    r = MATH_MAD(nf, -R_LOG2_BY_64_TL, MATH_MAD(nf, -R_LOG2_BY_64_LD, ylogx)) + ylogx_t;

    // Truncated Taylor series for e^r
    poly = MATH_MAD(MATH_MAD(MATH_MAD(r, 0x1.555556p-5f, 0x1.555556p-3f), r, 0x1.000000p-1f), r*r, r);

    tv = p_jby64[j];

    float expylnx = MATH_MAD(tv.s0, poly, MATH_MAD(tv.s1, poly, tv.s1)) + tv.s0;

    if (AMD_OPT()) {
        expylnx = BUILTIN_FLDEXP_F32(expylnx, m);
    } else {
        float sexpylnx = expylnx * AS_FLOAT(0x1 << (m + 149));
        float texpylnx = AS_FLOAT(AS_INT(expylnx) + (m << EXPSHIFTBITS_SP32));
        expylnx = m < -125 ? sexpylnx : texpylnx;
    }

    // Result is +-Inf if (ylogx + ylogx_t) > 128*log2
    expylnx = (ylogx > 0x1.62e430p+6f) | (ylogx == 0x1.62e430p+6f & ylogx_t > -0x1.05c610p-22f) ? AS_FLOAT(PINFBITPATT_SP32) : expylnx;

    // Result is 0 if ylogx < -149*log2
    expylnx = ylogx <  -0x1.9d1da0p+6f ? 0.0f : expylnx;

    // Classify y:
    //   inty = 0 means not an integer.
    //   inty = 1 means odd integer.
    //   inty = 2 means even integer.

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
    bool ax_eq_nan = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool ay_eq_0 = ay == 0.0f;
    bool ay_eq_pinf = BUILTIN_CLASS_F32(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F32(ay, CLASS_QNAN|CLASS_SNAN);
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
    bool ax_eq_nan = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN);
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
    bool ax_eq_nan = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN);
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
    bool ax_eq_nan = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN);
    bool x_pos = BUILTIN_CLASS_F32(x, CLASS_PZER|CLASS_PSUB|CLASS_PNOR|CLASS_PINF);
    bool x_eq_ninf = BUILTIN_CLASS_F32(x, CLASS_NINF);
    bool x_eq_pinf = BUILTIN_CLASS_F32(x, CLASS_PINF);
    bool ay_eq_0 = ay == 0.0f;
    bool ay_eq_pinf = BUILTIN_CLASS_F32(ay, CLASS_PINF);
    bool ay_eq_nan = BUILTIN_CLASS_F32(ay, CLASS_QNAN|CLASS_SNAN);
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

