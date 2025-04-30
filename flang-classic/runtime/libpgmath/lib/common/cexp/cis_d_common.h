
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "math_common.h"
#include "sleef_common.h"

/* 1152 bits of 1/PI for Payne-Hanek argument reduction. */
static uint64_t i1opi_f [] = {
    0x35fdafd88fc6ae84ULL,
    0x9e839cfbc5294975ULL,
    0xba93dd63f5f2f8bdULL,
    0xa7a31fb34f2ff516ULL,
    0xb69b3f6793e584dbULL,
    0xf79788c5ad05368fULL,
    0x8ffc4bffef02cc07ULL,
    0x4e422fc5defc941dULL,
    0x9cc8eb1cc1a99cfaULL,
    0x74ce38135a2fbf20ULL,
    0x74411afa975da242ULL,
    0x7f0ef58e5894d39fULL,
    0x0324977504e8c90eULL,
    0xdb92371d2126e970ULL,
    0xff28b1d5ef5de2b0ULL,
    0x6db14acc9e21c820ULL,
    0xfe13abe8fa9a6ee0ULL,
    0x517cc1b727220a94ULL,
    0ULL,
};

/* Payne-Hanek style argument reduction. */
static double
__reduction_slowpath_pio2(double const a, int64_t *h, double *rlo)
{
    // Compute bits of n = a * 1/pi using integer arithmetic.
    // No need to compute integer bits representing quotient 2
    // and above as trig functions are 2*pi periodic anyway.
    // Ultimately we'll be interested in 3 leading bits of n to
    // determine the octant into which a falls while the lower
    // bits will be used to compute a - n*pi that don't
    // cancel out in the subtraction. We'd like to obtain
    // the reduced argument to 10 bits more than available in
    // double precision.
    // We know thanks to Kahan that at most 60-61 bits may get
    // cancelled, so we'd need at most 61 + 53 + 10 = 124 bits
    // of the product, yet our choice of table storage leads
    // to a maximum of another 63 bits shift, so we will
    // end up computing 3*64 = 192 bits using 4*64 = 256 bits
    // of 1/pi.

    uint64_t result[4];
    // get sign bit
    uint64_t s = D2L(a) & DB_SIGN_BIT;
    // get unbiased exponent
    uint64_t e = ((D2L(a) >> 52) & 0x7ff) - DB_EXP_BIAS;
    // normalized unsigned integer representation of the
    // significand of a with implicit leading bit restored
    uint64_t ia = ((D2L(a) << 11) | 0x8000000000000000ULL);
    // index into the table of 1/pi bits based on exponent e
    // lookup table is organized as 19 words of 64-bit width
    int32_t idx = 15 - ((e >> 6) & 0xF);
    int32_t q;

    // multiply significand of a by bits of 1/pi
    __uint128_t acc = 0;
    for (q = 0; q < 4; q++) {
        // no integer wraparound here: given 3 p-bit integers
        // x*y+z doesn't exceed (2^p-1)*(2^p-1)+2^p-1 =
        // (2^p-1)*2^p, which is strictly less than 2^2p
        acc += (__uint128_t)ia * i1opi_f[idx + q];
        result[q] = (uint64_t)acc;
        acc >>= 64;
    }

    uint64_t p = result[3];
    // we avoided shifting or bit-addressing the lookup table,
    // by working with coarse 64-bit words, now need to fine-tune
    // the result
    e = e & 63;
    if (e) {
        p         = (p << e)         | (result[2] >> (64 - e));
        result[2] = (result[2] << e) | (result[1] >> (64 - e));
    }

    // p * pi may approach 1.1111...b * pi < 2*pi,
    // so its leading bit has cost of 2^(2) in the worst case.
    // Prepare double precision sign and exponent of the result
    uint64_t shi = s | (((uint64_t)(2 + DB_EXP_BIAS)) << 52);

    // 3 selector bits: pi, pi/2, pi/4
    *h = p & 0xE000000000000000ULL;
    // eliminate r > pi/4 by subtracting r from pi/2
    if (p & 0x2000000000000000ULL)
    {
        p = ~p;
        result[2] = ~result[2];
    }
    // discard selector bits
    p &= 0x1fffffffffffffffULL;

    // normalize p, adjust exponent bits accordingly
    int lz = __builtin_clzll(p);
    p = p << lz | result[2] >> (64 - lz);
    shi -= (uint64_t)lz << 52;

    // normalized p times normalized pi
    __uint128_t prod = p * (__uint128_t)0xc90fdaa22168c235ULL;
    // take 53 leading bits of the product and
    // convert to FP.
    uint64_t lhi = prod >> (64 + 11);
    // Leading bit in lhi may still be zero even though
    // we've normalized both p and pi, so we won't OR the
    // significant bits into FP mantissa, but will use
    // type conversion + multiply which will also take care of
    // this leading bit normalization.
    // convert 53-bit integer into high part
    double r = L2D(shi - (52ull << 52)) * (double)lhi;

    uint64_t llo = prod >> 11;
    // low part is at least 53 orders of magnitude
    // smaller than high, plus adjust scaling for
    // 64-bit integer conversion to FP.
    uint64_t slo = shi - (53ull << 52) - (63ull << 52);
    *rlo = L2D(slo) * (double)llo;

    return r;
}

static double _Complex __attribute__ ((noinline))
__cis_d_scalar(double x)
{
    // This function computes pair of sin/cos results per input and
    // relies on octant reduction: x -> r such that |r| < pi/4.
    // We then use minimax approximations for sin and cos.
    // It is possible to use quadrant reduction to be able to compute
    // only sin() and use it for both sin and cos reconstruction, but
    // we do not go that way because the reduced argument is larger
    // and requires more multiprecision in the polynomial computation
    // to not exceed our accuracy target.

    double p, w;
    int64_t h, selector;
    uint64_t ssign, csign;

    // p = |x|
    p = L2D(D2L(x) & DB_ABS_MASK);                                              PRINT(p);

#undef THRESHOLD
#define THRESHOLD L2D(0x42dce2c35b04bada)
#define DELTA 0.005
    if (__builtin_expect(p > THRESHOLD, 0))
    {
        // huge arguments reduction
        p = (D2L(p) >= DB_PINF) ? p * 0.0 : __reduction_slowpath_pio2(p, &h, &w); PRINT(p); PRINT(h); PRINT(w);
        // h contains: 3 selector bits for [pi, pi/2, extra swap]
        ssign = h & DB_SIGN_BIT;                                                PRINT(ssign);
        ssign = ssign ^ (D2L(x) & DB_SIGN_BIT);                                 PRINT(ssign);
        csign = ((h + (1ull<<62)) & DB_SIGN_BIT);                               PRINT(csign);
        selector = ((h << 1) + (h << 2)) >> 63;                                 PRINT(selector);
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

#undef _1_OVER_PI
#define _1_OVER_PI L2D(0x3fd45f306dc9c883)
        double k = fma(p, _1_OVER_PI*2.0, 0x1.8p52);                            PRINT(k);

        ssign = (D2L(k) & 0x2ull) << 62;                                        PRINT(ssign);
        ssign = ssign ^ (D2L(x) & DB_SIGN_BIT);                                 PRINT(ssign);
        csign = ((D2L(k) + 1) & 0x2ull) << 62;                                  PRINT(csign);
        //sign extend to obtain bitmask
        selector = 0 - (signed long long int)(D2L(k) & 0x1ull);                 PRINT(selector);
        k -= 0x1.8p52;                                                          PRINT(k);
#undef PI_HI
#undef PI_MI
#undef PI_LO
#define PI_HI L2D(0x400921fb54442d18)
#define PI_MI L2D(0x3ca1a62633145c00)
#define PI_LO L2D(0x398b839a252049c1)

        // first reduction step
        double u  = fma(k, -PI_HI/2.0, p);                                      PRINT(u);
        // second reduction step
        double v1 = fma(k, -PI_MI/2.0, u);                                      PRINT(v1);
        double p1, p2;
        fast2mul_dp(k, -PI_MI/2.0, &p1, &p2);                                   PRINT(p1); PRINT(p2);
        double t1, t2;
        fast2sum_dp(u, p1, &t1, &t2);                                           PRINT(t1); PRINT(t2);
        double v2 = (t1 - v1);
        v2 = v2 + t2;
        v2 = v2 + p2;                                                           PRINT(v1); PRINT(v2);
        // third reduction step
        w = fma(k, -PI_LO/2.0, v2);                                             PRINT(w);
        // now v1 + w give us the reduced argument in the double-double form
        // redistribute bits between high and low parts
        fast2sum_dp(v1, w, &p, &w);                                             PRINT(p); PRINT(w);
    }

    // assume that the range reduction produced argument
    // suitable for further minimax evaluation or NaN
    assert( (p != p) || ((fabs(p) / (PI_HI/4.0) - 1.0) < 2.0*DELTA) );

    double ps, pc;

    double twoWp = (2.0*w*p);
    double s = fma(p,p,twoWp);                                                  PRINT(s);

    // 2^(-55), interval: [+-pi/4*(1 + 2*delta)], delta=0.005
    #define S_1   L2D(0x3ff0000000000000)
    #define S_3   L2D(0xbfc5555555555554)
    #define S_5   L2D(0x3f81111111110a18)
    #define S_7   L2D(0xbf2a01a019e3c0a6)
    #define S_9   L2D(0x3ec71de37484189f)
    #define S_11  L2D(0xbe5ae5fc23541697)
    #define S_13  L2D(0x3de5df2f7b851b81)

    // 2^(-61), interval: [+-pi/4*(1 + 2*delta)], delta=0.005
    #define C_0   L2D(0x3ff0000000000000)
    #define C_2   L2D(0xbfe0000000000000)
    #define C_4   L2D(0x3fa5555555555552)
    #define C_6   L2D(0xbf56c16c16c15ee2)
    #define C_8   L2D(0x3efa01a019df3bcb)
    #define C_10  L2D(0xbe927e4f8e81f5e4)
    #define C_12  L2D(0x3e21eea7b7f9f23e)
    #define C_14  L2D(0xbda8ff5144c2ae4b)

    pc = C_14;
    pc = fma(pc, s, C_12);
    pc = fma(pc, s, C_10);
    pc = fma(pc, s, C_8);
    pc = fma(pc, s, C_6);
    pc = fma(pc, s, C_4);
    pc = fma(pc, s, C_2);
    pc = fma(pc, s, C_0);

    double p3 = s*p;
    ps = S_13;
    ps = fma(ps, s, S_11);
    ps = fma(ps, s, S_9);
    ps = fma(ps, s, S_7);
    ps = fma(ps, s, S_5);
    ps = fma(ps, s, S_3);
    ps = fma(ps, p3, w);
    ps += p;

    double rsin, rcos;
    rsin = L2D((selector & D2L(pc)) | ((~selector) & D2L(ps)));    PRINT(rsin);
    rsin = L2D(D2L(rsin) ^ ssign);                                 PRINT(rsin);
    rcos = L2D((selector & D2L(ps)) | ((~selector) & D2L(pc)));    PRINT(rcos);
    rcos = L2D(D2L(rcos) ^ csign);                                 PRINT(rcos);

    return set_cmplxd(rcos, rsin);
}

static vdouble __attribute__((noinline)) __vcis_d_slowpath(vdouble x, vopmask m)
{
    vdouble in = x;
    vdouble res = vSETd(0.0);
    int i;
    const int vlen = sizeof(vdouble) / sizeof(double _Complex);
    for (i = 0; i < vlen; i++)
    {
        // only work on even elements as odd ones are duplicates
        double a = *(2*i + (double *)(&in));
        // FIXME: here it might be more robust if we read the mask and decide,
        //        instead of duplicating the real condition
        if (__builtin_expect(!(L2D(D2L(a) & DB_ABS_MASK) > THRESHOLD), 0)) continue;
        *(0 + i + (double _Complex *)(&res)) = __cis_d_scalar(a);
    }
    return res;
}

//static vdouble __attribute((noinline))
static vdouble INLINE
__vcis_d_kernel(vdouble va)
{
    // This function will compute cos and sin in parallel
    // over the SIMD register filled with duplicate input
    // values: x x y y z z, etc
    // Will use the same range reduction for every value.
    // If we need to take huge reduction slow path, then
    // we will recall about duplicates.

    const vdouble vRS = vSETd(0x1.8p52);
    vdouble vabsa = vL2D(vand_vi2_vi2_vi2(vD2L(va), vSETll(DB_ABS_MASK)));      PRINT(vabsa);
    vdouble k = vfma_vd_vd_vd_vd(vabsa, vSETd(_1_OVER_PI * 2.0), vRS);          PRINT(k);
    // Need k in sin positions and k+1 in cos positions
    const vint2 _0_1 = vSETLLL(0, 1);
    // cssign merged sign
    vint2  cssign = vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(vD2L(k), _0_1), vSETll(2));
           cssign = vsll64_vi2_vi2_i(cssign, 62);
           cssign = vxor_vi2_vi2_vi2(cssign, vand_vi2_vi2_vi2(vD2L(va), vSETLLL(DB_SIGN_BIT, 0)));
    // parity bit of k is the selector
    vopmask selector = veq64_vo_vm_vm(vand_vi2_vi2_vi2(vD2L(k), vSETll(1)), vSETll(1));
           // k rounded to integer, now in normalized FP format
           k = vsub_vd_vd_vd(k, vRS);                                           PRINT(k);
    // first reduction step
    vdouble u = vfma_vd_vd_vd_vd(k, vSETd(-PI_HI/2.0), vabsa);                  PRINT(u);
    // second reduction step
    vdouble v1 = vfma_vd_vd_vd_vd(k,  vSETd(-PI_MI/2.0), u);                    PRINT(v1);
    vdouble p1, p2;
            vfast2mul_dp(k, vSETd(-PI_MI/2.0), &p1, &p2);                       PRINT(p1); PRINT(p2);
    vdouble t1, t2;
            vfast2sum_dp(u, p1, &t1, &t2);
    vdouble v2 = vsub_vd_vd_vd(t1, v1);
            v2 = vadd_vd_vd_vd(v2, t2);
            v2 = vadd_vd_vd_vd(v2, p2);
    // third reduction step
    vdouble w = vfma_vd_vd_vd_vd(k, vSETd(-PI_LO/2.0), v2);
    vdouble p;
    // now v1 + w give us the reduced argument in the double-double form
    // redistribute bits between high and low parts
            vfast2sum_dp(v1, w, &p, &w);
    vdouble x2 = vfma_vd_vd_vd_vd(p, p, vmul_vd_vd_vd(vmul_vd_vd_vd(p, w), vSETd(2.0)));

    // compute polynomials: minimax on +-pi/4.0*1.01
                                 //sin coef, cos coef
    vdouble c14s13 = vL2D(vSETLLL(D2L(S_13), D2L(C_14)));
    vdouble c12s11 = vL2D(vSETLLL(D2L(S_11), D2L(C_12)));
    vdouble c10s9  = vL2D(vSETLLL(D2L( S_9), D2L(C_10)));
    vdouble c8s7   = vL2D(vSETLLL(D2L( S_7), D2L( C_8)));
    vdouble c6s5   = vL2D(vSETLLL(D2L( S_5), D2L( C_6)));
    vdouble c4s3   = vL2D(vSETLLL(D2L( S_3), D2L( C_4)));

    vdouble x2_0 = vL2D(vand_vi2_vi2_vi2(vD2L(x2), vSETLLL(0, -1LL)));
    vdouble x2_1 = vL2D(vor_vi2_vi2_vi2(vD2L(x2_0), vSETLLL(DB_ONE, 0)));
    vdouble  _0p = vL2D(vand_vi2_vi2_vi2(vD2L(p), vSETLLL(-1LL, 0)));
    vdouble  _0w = vL2D(vand_vi2_vi2_vi2(vD2L(w), vSETLLL(-1LL, 0)));
    vdouble  _1p = vL2D(vor_vi2_vi2_vi2(vD2L(_0p),vSETLLL(0, DB_ONE)));
    vdouble  c2w = vL2D(vor_vi2_vi2_vi2(vD2L(_0w),vSETLLL(0, D2L(C_2))));
    vdouble x2x3 = vmul_vd_vd_vd(x2, _1p);

    vdouble pcps = c14s13;
           pcps = vfma_vd_vd_vd_vd(pcps, x2, c12s11);
           pcps = vfma_vd_vd_vd_vd(pcps, x2, c10s9);
           pcps = vfma_vd_vd_vd_vd(pcps, x2, c8s7);
           pcps = vfma_vd_vd_vd_vd(pcps, x2, c6s5);
           pcps = vfma_vd_vd_vd_vd(pcps, x2, c4s3);
           pcps = vfma_vd_vd_vd_vd(pcps, x2x3, c2w);
           pcps = vfma_vd_vd_vd_vd(pcps, x2_1, _1p);

    // swap cos and sin in SIMD register
    vdouble pspc = vrev21_vd_vd(pcps);
    // select cos or sin
    vdouble vres = vsel_vd_vo_vd_vd(selector, pspc, pcps);
            // fixup sign
            vres = vL2D(vxor_vi2_vi2_vi2(vD2L(vres), cssign));                  PRINT(vres);

    vopmask m = vgt_vo_vd_vd(vabsa, vSETd(THRESHOLD));                          PRINT(m);
    if (__builtin_expect(!vtestz_i_vo(m), 0))
    {
        // blend slow and fast path results
        vdouble vslowfres = __vcis_d_slowpath(va, m);
        vres = vsel_vd_vo_vd_vd(m, vslowfres, vres);
    }

    return vres;
}
