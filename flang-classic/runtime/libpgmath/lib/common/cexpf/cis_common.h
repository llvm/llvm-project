
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "common.h"
#include "math_common.h"
#include "sleef_common.h"

/* Payne-Hanek style argument reduction. */
static float
__attribute__ ((noinline))
//__attribute__ ((always_inline)) inline
__reduction_slowpath_pio2(float const a, int32_t *h, float *w)
{
    uint2 m;
    uint32_t ia = float_as_int(a);
    uint32_t result[7];
    uint32_t hi, lo;
    uint32_t e;
    int32_t idx;
    int32_t q;

    *w = 0.0f;
    if ((unsigned)float_as_int(a) >= 0x7f800000) return a * 0.0f;

    e = ((ia >> 23) & 0xff) - 127;
    ia = (ia << 8) | 0x80000000;

    /* compute x * 1/pi */
    idx = 4 - ((e >> 5) & 3);

    hi = 0;
    for (q = 0; q < 6; q++) {
        m = umad32wide(i1opi_f[q], ia, hi);
        lo = m.x;
        hi = m.y;
        result[q] = lo;
    }
    result[q] = hi;

    e = e & 31;
    /* shift result such that hi:lo<63:63> is the least significant
       integer bit, and hi:lo<62:0> are the fractional bits of the result
    */

    uint64_t p;
    p = (uint64_t)result[idx + 2] << 32;
    p |= result[idx + 1];
    // 32-int right shift by >=32 is undefined, so need to
    // promote to 64 bits first
    p = (p << e) | ((uint64_t)(result[idx]) >> (32 - e));

    /* fraction */
    q  = (p >> 32) & 0xE0000000; //3 selector bits

    // eliminate r > pi/4 by subtracting r from pi/2
    if (p & 0x2000000000000000ULL)
    {
        p = ~p;
    }
    // discard selector bits
    p &= 0x1fffffffffffffffULL;

    *h = q;
    double d = (double)(int64_t)p;
    d *= PI_2_M63;

    float r = (float)d;
    *w = (float)(d - (double)r);

    return r;
}

//__attribute__ ((noinline))
static float _Complex __attribute__((always_inline)) inline
__cis_scalar_kernel(float x)
{
    float p, k, r, s, t, w;
    float x2, x3, ps, pc;
    int h = 0;
    int sign = 0, csign = 0, ssign = 0;
    int selector = 0;

    // p = |x|
    p = I2F(F2I(x) & FL_ABS_MASK);                                 PRINT(p);
#undef THRESHOLD_F
#define THRESHOLD_F 0x1.7c3e58p17
#define DELTA 0.005
    if (__builtin_expect(p > THRESHOLD_F, 0))
    {
        // huge arguments reduction
        p = __reduction_slowpath_pio2(p, &h, &w); PRINT(p); PRINT(h); PRINT(w);
        // h contains: selector bits for [pi, pi/2, extra swap]
        ssign = h & FL_SIGN_BIT;                                   PRINT(ssign);
        ssign = ssign ^ (F2I(x) & FL_SIGN_BIT);                    PRINT(ssign);
        csign = ((h + (1<<30)) & FL_SIGN_BIT);                     PRINT(csign);
        selector = ((h << 1) + (h << 2)) >> 31;                    PRINT(selector);
    } else {
        // Here we follow the reduction process presented in
        // [1] S. Boldo, M. Daumas & R-C. Li,
        // Formally Verified Argument Reduction with a Fused-Multiply-Add,
        // IEEE Transactions on Computers (IF: 1.680), 58(8), pp. 1139-1145, 2009.
        // i.e. we compute (x - N*pi/2) using 3-double representation of
        // pi/2 and obtain result as an unevaluated sum of two doubles: p + w.

        // NOTE: this process becomes inaccurate because of the first step
        // where we compute x / (pi/2) via multiplication of x by R,
        // rounded 2/pi. So when x grows that rounding error in the 2/pi
        // may propagate into integer part of the quotient and the resulting
        // reduced argument will fall out of desired [-pi/4, pi/4] range.
        // Here's what happens:
        //     R = 2/pi + eps
        //     where eps is an absolute error in the rounded constant.
        // We may always mathematically represent x as
        //     x = N*pi/2 + m*pi/2
        // with N being nearest integer and m < 1/2.
        // Now the multiplication x*R results in:
        //     x*R = x*2/pi + x*eps = N + m + x*eps
        // In order for this to round correctly to nearest integer N
        // x*eps shall be small enough, so that it doesn't bump m over 1/2.
        // Suppose m = 1/2 - delta, delta is positive since m < 1/2.
        // x*eps must be less than delta in order for the rounding to be
        // correct and result in N.
        // eps is known to us by definition of R. Acceptable range for the
        // input |x| < T is given to us by the paper [1]. So the error will
        // appear when T*eps > delta.
        // For T = pi/2 * (2^(p-2) - 1) and eps ~= 2/pi*2^(-p-1) we get
        //     T*eps ~= 1/8
        // Now it is easy to find inputs around integer
        // multiples of pi/16 that will have delta <= 1/8:
        //     e.g. N*pi/2 + 1/2*pi/2 - 1/8 * pi/2
        // So what happens when the rounding goes wrong by 1? The reduction
        //     r = x - (N+1)*pi/2 = m*pi/2 - pi/2 = pi/2*(m - 1)
        //       = -pi/2*(1/2 + delta) = -pi/4*(1 + 2*delta)
        // produces reduced argument r, which is off from the target interval
        // proportionally to 2*delta. Approximation error for the minimax
        // polynomial oscillates near the interval bounds and
        // grows significantly outside the interval, so we may end up with
        // computed results that will be off by too much.
        // We may also notice that small delta doesn't actually pose the same
        // threat as the large delta in terms of being off from the
        // approximation interval, so we are not interested in finding
        // the lower limit on delta in the given precision.
        // Instead we accept that the problem with reduction comes from the
        // arguments 'not too close' to multiples of pi/4. And the larger
        // |x| is, the larger x*eps becomes and thus we need to accomodate
        // for larger deltas. We may reduce the effect of the interval error
        // by allowing our approximation interval to be wider by some
        // margin, perhaps grow it until we need to add an extra degree to
        // the polynomial.
        // But we also do not want delta to be too large since the
        // reduced argument may end up in a different binade and that would
        // affect the polynomial evaluation error: instead of reducing
        // to pi/4 < 2^0 we may end up with r >= 2^0.
        // With bad delta capped by DELTA from the choice of
        // approximation interval bounds:
        //     pi/4*(1 + 2*DELTA)
        // we shall work from the other end and ensure x*eps < DELTA
        // by lowering the threshold for |x| < min(T, DELTA/eps).

#undef _1_OVER_PI_F
#define _1_OVER_PI_F I2F(0x3ea2f983)
        k = fmaf(p, _1_OVER_PI_F * 2.0f, 0x1.8p23);                PRINT(k);
        ssign = (F2I(k) & 0x2) << 30;                              PRINT(ssign);
        ssign = ssign ^ (F2I(x) & FL_SIGN_BIT);                    PRINT(ssign);
        csign = ((float_as_int(k) + 1) & 0x2) << 30;               PRINT(csign);
        //sign extend to obtain bitmask
        selector = 0 - (float_as_int(k) & 0x1);                    PRINT(selector);
        k -= 0x1.8p23;                                             PRINT(k);
#undef  PI_HI_F
#undef  PI_MI_F
#undef  PI_LO_F
#define PI_HI_F I2F(0x40490fdc)
#define PI_MI_F I2F(0xb4aeef48)
#define PI_LO_F I2F(0xa9e7b967)

        // first reduction step
        float u  = fmaf(k, -PI_HI_F/2.0f, p);                      PRINT(u);
        // second reduction step
        float v1 = fmaf(k, -PI_MI_F/2.0f, u);                      PRINT(v1);
        float p1, p2;
        fast2mul(k, -PI_MI_F/2.0f, &p1, &p2);                      PRINT(p1); PRINT(p2);
        float t1, t2;
        fast2sum(u, p1, &t1, &t2);                                 PRINT(t1); PRINT(t2);
        float v2 = (t1 - v1);
        v2 = v2 + t2;
        v2 = v2 + p2;                                              PRINT(v1); PRINT(v2);
        // third reduction step
        w = fmaf(k, -PI_LO_F/2.0f, v2);                            PRINT(w);
        // now v1 + w give us the reduced argument in the double-float form
        // redistribute bits between high and low parts
        fast2sum(v1, w, &p, &w);                                   PRINT(p); PRINT(w);
    }

    // assume that the range reduction produced argument
    // suitable for further minimax evaluation or NaN
    assert( (p != p) || ((fabsf(p) / (PI_HI_F/4.0f) - 1.0f) < 2.0f*DELTA) );

    x2 = fmaf(p,p, (2.0f*w*p));                                    PRINT(x2);

// 2^(-26), interval: [+-pi/4*(1 + 2*delta)], delta=0.005
#define S1 I2F(0x3f800000)
#define S3 I2F(0xbe2aaaa9)
#define S5 I2F(0x3c0885e1)
#define S7 I2F(0xb94d4ec3)

// 2^(-31), interval: [+-pi/4*(1 + 2*delta)], delta=0.005
#define C0 I2F(0x3f800000)
#define C2 I2F(0xbf000000)
#define C4 I2F(0x3d2aaaa7)
#define C6 I2F(0xbab60727)
#define C8 I2F(0x37cd39b8)

    ps = S7;
    ps = fmaf(ps, x2, S5);
    ps = fmaf(ps, x2, S3);
    x3 = p * x2;
    ps = fmaf(ps, x3, w);                                          PRINT(ps);
    ps = ps + p;                                                   PRINT(ps);

    pc = C8;
    pc = fmaf(pc, x2, C6);
    pc = fmaf(pc, x2, C4);
    pc = fmaf(pc, x2, C2);
    pc = fmaf(pc, x2, C0);                                         PRINT(pc);

    float rsin, rcos;
    rsin = I2F((selector & F2I(pc)) | ((~selector) & F2I(ps)));    PRINT(rsin);
    rsin = I2F(F2I(rsin) ^ ssign);                                 PRINT(rsin);
    rcos = I2F((selector & F2I(ps)) | ((~selector) & F2I(pc)));    PRINT(rcos);
    rcos = I2F(F2I(rcos) ^ csign);                                 PRINT(rcos);

    return set_cmplx(rcos, rsin);
}

static vfloat __attribute__((noinline)) __vcis_slowpath(vfloat x, vopmask m)
{
    vfloat in = x;
    vfloat res = vSETf(0.0f);
    int i;
    const int vlen = sizeof(vfloat) / sizeof(float _Complex);
    for (i = 0; i < vlen; i++)
    {
        // only work on even elements as odd ones are duplicates
        float a = *(2*i + (float *)(&in));
        // FIXME: here it might be more robust if we read the mask and decide,
        //        instead of duplicating the real condition
        if (__builtin_expect(!(I2F(F2I(a) & FL_ABS_MASK) > THRESHOLD_F), 0)) continue;
        *(i + (float _Complex *)(&res)) = __cis_scalar_kernel(a);
    }
    return res;
}

//static vfloat __attribute((noinline))
static vfloat INLINE
__vcis_kernel(vfloat va)
{
    // This function will compute cos and sin in parallel
    // over the SIMD register filled with duplicate input
    // values: x x y y z z, etc
    // Will use the same range reduction for every value.
    // If we need to take huge reduction slow path, then
    // we will recall about duplicates.

    const vfloat vRS = vSETf(0x1.8p23);
    vfloat vabsa = vI2F(vand_vi2_vi2_vi2(vF2I(va), vSETi(FL_ABS_MASK)));        PRINT(vabsa);
    vfloat k = vfma_vf_vf_vf_vf(vabsa, vSETf(_1_OVER_PI_F * 2.0f), vRS);        PRINT(k);
    // Need k in sin positions and k+1 in cos positions
    const vint2 _0_1 = vSETLLi(0x00000000, 0x00000001);
    // cssign merged sign
    vint2  cssign = vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(vF2I(k), _0_1), vSETi(0x00000002));
           cssign = vsll_vi2_vi2_i(cssign, 30);
           cssign = vxor_vi2_vi2_vi2(cssign, vand_vi2_vi2_vi2(vF2I(va), vSETLLi(FL_SIGN_BIT, 0x00000000)));
    // parity bit of k is the selector
    vopmask selector = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vF2I(k), vSETi(1)), vSETi(1));
           // k rounded to integer, now in normalized FP format
           k = vsub_vf_vf_vf(k, vRS);                                           PRINT(k);
    // first reduction step
    vfloat u = vfma_vf_vf_vf_vf(k, vSETf(-PI_HI_F/2.0f), vabsa);                PRINT(u);
    // second reduction step
    vfloat v1 = vfma_vf_vf_vf_vf(k,  vSETf(-PI_MI_F/2.0f), u);                  PRINT(v1);
    vfloat p1, p2;
           vfast2mul(k, vSETf(-PI_MI_F/2.0f), &p1, &p2);                        PRINT(p1); PRINT(p2);
    vfloat t1, t2;
           vfast2sum(u, p1, &t1, &t2);
    vfloat v2 = vsub_vf_vf_vf(t1, v1);
           v2 = vadd_vf_vf_vf(v2, t2);
           v2 = vadd_vf_vf_vf(v2, p2);
    // third reduction step
    vfloat w = vfma_vf_vf_vf_vf(k, vSETf(-PI_LO_F/2.0f), v2);
    vfloat p;
    // now v1 + w give us the reduced argument in the double-float form
    // redistribute bits between high and low parts
           vfast2sum(v1, w, &p, &w);
    vfloat x2 = vfma_vf_vf_vf_vf(p, p, vmul_vf_vf_vf(vmul_vf_vf_vf(p, w), vSETf(2.0f)));

    // compute polynomials: minimax on +-pi/4.0*1.01
                       //sin coef, cos coef
    vfloat c8s7 = vI2F(vSETLLi(F2I(S7), F2I(C8)));
    vfloat c6s5 = vI2F(vSETLLi(F2I(S5), F2I(C6)));
    vfloat c4s3 = vI2F(vSETLLi(F2I(S3), F2I(C4)));

    vfloat x2_0 = vI2F(vand_vi2_vi2_vi2(vF2I(x2),vSETLLi(0x00000000, 0xffffffff)));
    vfloat x2_1 = vI2F(vor_vi2_vi2_vi2(vF2I(x2_0),vSETLLi(FL_ONE,    0x00000000)));
    vfloat  _0p = vI2F(vand_vi2_vi2_vi2(vF2I(p), vSETLLi(0xffffffff, 0x00000000)));
    vfloat  _0w = vI2F(vand_vi2_vi2_vi2(vF2I(w), vSETLLi(0xffffffff, 0x00000000)));
    vfloat  _1p = vI2F(vor_vi2_vi2_vi2(vF2I(_0p),vSETLLi(0x00000000, FL_ONE)));
    vfloat  c2w = vI2F(vor_vi2_vi2_vi2(vF2I(_0w),vSETLLi(0x00000000, F2I(C2))));
    vfloat x2x3 = vmul_vf_vf_vf(x2, _1p);

    vfloat pcps = c8s7;
           pcps = vfma_vf_vf_vf_vf(pcps, x2, c6s5);
           pcps = vfma_vf_vf_vf_vf(pcps, x2, c4s3);
           pcps = vfma_vf_vf_vf_vf(pcps, x2x3, c2w);
           pcps = vfma_vf_vf_vf_vf(pcps, x2_1, _1p);

    // swap cos and sin in SIMD register
    vfloat pspc = vrev21_vf_vf(pcps);
    // select cos or sin
    vfloat vres = vsel_vf_vo_vf_vf(selector, pspc, pcps);
           // fixup sign
           vres = vI2F(vxor_vi2_vi2_vi2(vF2I(vres), cssign));                   PRINT(vres);

    vopmask m = vgt_vo_vf_vf(vabsa, vSETf(THRESHOLD_F));                        PRINT(m);
    if (__builtin_expect(!vtestz_i_vo(m), 0))
    {
        // blend slow and fast path results
        vfloat vslowfres = __vcis_slowpath(va, m);
        vres = vsel_vf_vo_vf_vf(m, vslowfres, vres);
    }

    return vres;
}
