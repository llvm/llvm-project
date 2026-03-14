//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes pow using log and exp
//
//   x^y = exp(y * log(x))
//
// We take care not to lose precision in the intermediate steps.
//
// When computing log, calculate it in splits:
//
//   r = f * (p_invead + p_inv_tail)
//   r = rh + rt
//
// Calculate log polynomial using r, in end addition, do:
//
//   poly = poly + ((rh-r) + rt)
//
//   lth = -r
//   ltt = ((xexp * log2_t) - poly) + logT
//   lt = lth + ltt
//
//   lh = (xexp * log2_h) + logH
//   l = lh + lt
//
// Calculate final log answer as gh and gt:
//
//   gh = l & higher-half bits
//   gt = (((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh))
//
//   yh = y & higher-half bits
//   yt = y - yh
//
// Before entering computation of exp:
//
//   vs = ((yt*gt + yt*gh) + yh*gt)
//   v = vs + yh*gh
//   vt = ((yh*gh - v) + vs)
//
// In calculation of exp, add vt to r that is used for poly.
//
// At the end of exp, do:
//
//   ((((expT * poly) + expT) + expH*poly) + expH)
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_rootn(__CLC_GENTYPE x,
                                                 __CLC_INTN ny) {
  __CLC_GENTYPE y = MATH_RECIP(__CLC_CONVERT_GENTYPE(ny));

  __CLC_INTN ix = __CLC_AS_INTN(x);
  __CLC_INTN ax = ix & EXSIGNBIT_SP32;
  __CLC_INTN xpos = ix == ax;

  __CLC_INTN iy = __CLC_AS_INTN(y);
  __CLC_INTN ay = iy & EXSIGNBIT_SP32;
  __CLC_INTN ypos = iy == ay;

  // Extra precise log calculation
  // First handle case that x is close to 1
  __CLC_GENTYPE r = 1.0f - __CLC_AS_GENTYPE(ax);
  __CLC_INTN near1 = __clc_fabs(r) < 0x1.0p-4f;
  __CLC_GENTYPE r2 = r * r;

  // Coefficients are just 1/3, 1/4, 1/5 and 1/6
  __CLC_GENTYPE poly = __clc_mad(
      r,
      __clc_mad(r,
                __clc_mad(r, __clc_mad(r, 0x1.24924ap-3f, 0x1.555556p-3f),
                          0x1.99999ap-3f),
                0x1.000000p-2f),
      0x1.555556p-2f);

  poly *= r2 * r;

  __CLC_GENTYPE lth_near1 = -r2 * 0.5f;
  __CLC_GENTYPE ltt_near1 = -poly;
  __CLC_GENTYPE lt_near1 = lth_near1 + ltt_near1;
  __CLC_GENTYPE lh_near1 = -r;
  __CLC_GENTYPE l_near1 = lh_near1 + lt_near1;

  // Computations for x not near 1
  __CLC_INTN m = __CLC_CONVERT_INTN(ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
  __CLC_GENTYPE mf = __CLC_CONVERT_GENTYPE(m);
  __CLC_INTN ixs = __CLC_AS_INTN(__CLC_AS_GENTYPE(ax | 0x3f800000) - 1.0f);
  __CLC_GENTYPE mfs = __CLC_CONVERT_GENTYPE((ixs >> EXPSHIFTBITS_SP32) - 253);
  __CLC_INTN c = m == -127;
  __CLC_INTN ixn = c ? ixs : ax;
  __CLC_GENTYPE mfn = c ? mfs : mf;

  __CLC_INTN indx = (ixn & 0x007f0000) + ((ixn & 0x00008000) << 1);

  // F - Y
  __CLC_GENTYPE f = __CLC_AS_GENTYPE(0x3f000000 | indx) -
                    __CLC_AS_GENTYPE(0x3f000000 | (ixn & MANTBITS_SP32));

  indx = indx >> 16;
  __CLC_GENTYPE rh = f * __CLC_USE_TABLE(log_inv_tbl_ep_head, indx);
  __CLC_GENTYPE rt = f * __CLC_USE_TABLE(log_inv_tbl_ep_tail, indx);
  ;
  r = rh + rt;

  poly = __clc_mad(r, __clc_mad(r, 0x1.0p-2f, 0x1.555556p-2f), 0x1.0p-1f) *
         (r * r);
  poly += (rh - r) + rt;

  const __CLC_GENTYPE LOG2_HEAD = 0x1.62e000p-1f;  // 0.693115234
  const __CLC_GENTYPE LOG2_TAIL = 0x1.0bfbe8p-15f; // 0.0000319461833
  __CLC_GENTYPE lth = -r;
  __CLC_GENTYPE ltt =
      __clc_mad(mfn, LOG2_TAIL, -poly) + __CLC_USE_TABLE(loge_tbl_hi, indx);
  __CLC_GENTYPE lt = lth + ltt;
  __CLC_GENTYPE lh =
      __clc_mad(mfn, LOG2_HEAD, __CLC_USE_TABLE(loge_tbl_lo, indx));
  __CLC_GENTYPE l = lh + lt;

  // Select near 1 or not
  lth = near1 ? lth_near1 : lth;
  ltt = near1 ? ltt_near1 : ltt;
  lt = near1 ? lt_near1 : lt;
  lh = near1 ? lh_near1 : lh;
  l = near1 ? l_near1 : l;

  __CLC_GENTYPE gh = __CLC_AS_GENTYPE(__CLC_AS_UINTN(l) & 0xfffff000);
  __CLC_GENTYPE gt = ((ltt - (lt - lth)) + ((lh - l) + lt)) + (l - gh);

  __CLC_GENTYPE yh = __CLC_AS_GENTYPE(__CLC_AS_UINTN(iy) & 0xfffff000);

  __CLC_GENTYPE fny = __CLC_CONVERT_GENTYPE(ny);
  __CLC_GENTYPE fnyh = __CLC_AS_GENTYPE(__CLC_AS_UINTN(fny) & 0xfffff000);
  __CLC_GENTYPE fnyt = __CLC_CONVERT_GENTYPE(ny - __CLC_CONVERT_INTN(fnyh));
  __CLC_GENTYPE yt =
      MATH_DIVIDE(__clc_mad(-fnyt, yh, __clc_mad(-fnyh, yh, 1.0f)), fny);

  __CLC_GENTYPE ylogx_s = __clc_mad(gt, yh, __clc_mad(gh, yt, yt * gt));
  __CLC_GENTYPE ylogx = __clc_mad(yh, gh, ylogx_s);
  __CLC_GENTYPE ylogx_t = __clc_mad(yh, gh, -ylogx) + ylogx_s;

  // Extra precise exp of ylogx
  const __CLC_GENTYPE R_64_BY_LOG2 =
      0x1.715476p+6f; // 64/log2 : 92.332482616893657
  __CLC_INTN n = __CLC_CONVERT_INTN(ylogx * R_64_BY_LOG2);
  __CLC_GENTYPE nf = __CLC_CONVERT_GENTYPE(n);

  __CLC_INTN j = n & 0x3f;
  m = n >> 6;
  __CLC_INTN m2 = m << EXPSHIFTBITS_SP32;

  // log2/64 lead: 0.0108032227
  const __CLC_GENTYPE R_LOG2_BY_64_LD = 0x1.620000p-7f;
  // log2/64 tail: 0.0000272020388
  const __CLC_GENTYPE R_LOG2_BY_64_TL = 0x1.c85fdep-16f;
  r = __clc_mad(nf, -R_LOG2_BY_64_TL, __clc_mad(nf, -R_LOG2_BY_64_LD, ylogx)) +
      ylogx_t;

  // Truncated Taylor series for e^r
  poly = __clc_mad(__clc_mad(__clc_mad(r, 0x1.555556p-5f, 0x1.555556p-3f), r,
                             0x1.000000p-1f),
                   r * r, r);

  __CLC_GENTYPE exph = __CLC_USE_TABLE(exp_tbl_ep_head, j);
  __CLC_GENTYPE expt = __CLC_USE_TABLE(exp_tbl_ep_tail, j);

  __CLC_GENTYPE expylogx =
      __clc_mad(exph, poly, __clc_mad(expt, poly, expt)) + exph;
  __CLC_GENTYPE sexpylogx =
      __clc_fp32_subnormals_supported()
          ? expylogx * __CLC_AS_GENTYPE((__CLC_INTN)0x1 << (m + 149))
          : 0.0f;

  __CLC_GENTYPE texpylogx = __CLC_AS_GENTYPE(__CLC_AS_INTN(expylogx) + m2);
  expylogx = m < -125 ? sexpylogx : texpylogx;

  // Result is +-Inf if (ylogx + ylogx_t) > 128*log2
  expylogx = ((ylogx > 0x1.62e430p+6f) |
              (ylogx == 0x1.62e430p+6f & ylogx_t > -0x1.05c610p-22f))
                 ? __CLC_AS_GENTYPE((__CLC_UINTN)PINFBITPATT_SP32)
                 : expylogx;

  // Result is 0 if ylogx < -149*log2
  expylogx = ylogx < -0x1.9d1da0p+6f ? 0.0f : expylogx;

  // Classify y:
  //   inty = 0 means not an integer.
  //   inty = 1 means odd integer.
  //   inty = 2 means even integer.

  __CLC_INTN inty = 2 - (ny & 1);

  __CLC_GENTYPE signval =
      __CLC_AS_GENTYPE((__CLC_AS_UINTN(expylogx) ^ SIGNBIT_SP32));
  expylogx = ((inty == 1) & !xpos) ? signval : expylogx;
  __CLC_INTN ret = __CLC_AS_INTN(expylogx);

  // Corner case handling
  __CLC_BIT_INTN x_is_ninf = ix == (__CLC_INTN)NINFBITPATT_SP32;
  __CLC_BIT_INTN x_is_pinf = ix == (__CLC_INTN)PINFBITPATT_SP32;

  ret = (!xpos & (inty == 2)) ? __CLC_AS_INTN(__CLC_GENTYPE_NAN) : ret;
  __CLC_INTN xinf =
      xpos ? (__CLC_INTN)PINFBITPATT_SP32 : (__CLC_INTN)NINFBITPATT_SP32;
  ret = ((ax == 0) & !ypos & (inty == 1)) ? xinf : ret;
  ret = ((ax == 0) & !ypos & (inty == 2)) ? PINFBITPATT_SP32 : ret;
  ret = ((ax == 0) & ypos & (inty == 2)) ? 0 : ret;
  __CLC_INTN xzero = xpos ? 0 : (__CLC_INTN)0x80000000;
  ret = ((ax == 0) & ypos & (inty == 1)) ? xzero : ret;
  ret = (x_is_ninf & ypos & (inty == 1)) ? (__CLC_INTN)NINFBITPATT_SP32 : ret;
  ret = (x_is_ninf & !ypos & (inty == 1)) ? (__CLC_INTN)0x80000000 : ret;
  ret = (x_is_pinf & !ypos) ? 0 : ret;
  ret = (x_is_pinf & ypos) ? PINFBITPATT_SP32 : ret;
  ret = ax > PINFBITPATT_SP32 ? ix : ret;
  ret = ny == 0 ? __CLC_AS_INTN(__CLC_GENTYPE_NAN) : ret;

  return __CLC_AS_GENTYPE(ret);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_rootn(__CLC_GENTYPE x,
                                                 __CLC_INTN ny) {
  const __CLC_GENTYPE real_log2_tail = 5.76999904754328540596e-08;
  const __CLC_GENTYPE real_log2_lead = 6.93147122859954833984e-01;

  __CLC_GENTYPE dny = __CLC_CONVERT_GENTYPE(ny);
  __CLC_GENTYPE y = 1.0 / dny;

  __CLC_LONGN ux = __CLC_AS_LONGN(x);
  __CLC_LONGN ax = __CLC_AS_LONGN(__clc_fabs(x));
  __CLC_BIT_INTN xpos = ax == ux;

  __CLC_LONGN uy = __CLC_AS_LONGN(y);
  __CLC_LONGN ay = __CLC_AS_LONGN(__clc_fabs(y));
  __CLC_BIT_INTN ypos = ay == uy;

  // Extended precision log
  __CLC_GENTYPE v, vt;
  {
    __CLC_INTN exp = __CLC_CONVERT_INTN(ax >> 52) - 1023;
    __CLC_INTN mask_exp_1023 = exp == -1023;
    __CLC_GENTYPE xexp = __CLC_CONVERT_GENTYPE(exp);
    __CLC_LONGN mantissa = ax & 0x000FFFFFFFFFFFFFL;

    __CLC_LONGN temp_ux =
        __CLC_AS_LONGN(__CLC_AS_GENTYPE(0x3ff0000000000000L | mantissa) - 1.0);
    exp = __CLC_CONVERT_INTN((temp_ux & 0x7FF0000000000000L) >> 52) - 2045;
    __CLC_GENTYPE xexp1 = __CLC_CONVERT_GENTYPE(exp);
    __CLC_LONGN mantissa1 = temp_ux & 0x000FFFFFFFFFFFFFL;

    xexp = __CLC_CONVERT_LONGN(mask_exp_1023) ? xexp1 : xexp;
    mantissa = __CLC_CONVERT_LONGN(mask_exp_1023) ? mantissa1 : mantissa;

    __CLC_LONGN rax = (mantissa & 0x000ff00000000000) +
                      ((mantissa & 0x0000080000000000) << 1);
    __CLC_INTN index = __CLC_CONVERT_INTN(rax >> 44);

    __CLC_GENTYPE F = __CLC_AS_GENTYPE(rax | 0x3FE0000000000000L);
    __CLC_GENTYPE Y = __CLC_AS_GENTYPE(mantissa | 0x3FE0000000000000L);
    __CLC_GENTYPE f = F - Y;
    __CLC_GENTYPE log_h = __CLC_USE_TABLE(log_f_inv_tbl_head, index);
    __CLC_GENTYPE log_t = __CLC_USE_TABLE(log_f_inv_tbl_tail, index);
    __CLC_GENTYPE f_inv = (log_h + log_t) * f;
    __CLC_GENTYPE r1 =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(f_inv) & 0xfffffffff8000000L);
    __CLC_GENTYPE r2 = __clc_fma(-F, r1, f) * (log_h + log_t);
    __CLC_GENTYPE r = r1 + r2;

    __CLC_GENTYPE poly = __clc_fma(
        r,
        __clc_fma(r,
                  __clc_fma(r, __clc_fma(r, 1.0 / 7.0, 1.0 / 6.0), 1.0 / 5.0),
                  1.0 / 4.0),
        1.0 / 3.0);
    poly = poly * r * r * r;

    __CLC_GENTYPE hr1r1 = 0.5 * r1 * r1;
    __CLC_GENTYPE poly0h = r1 + hr1r1;
    __CLC_GENTYPE poly0t = r1 - poly0h + hr1r1;
    poly = __clc_fma(r1, r2, __clc_fma(0.5 * r2, r2, poly)) + r2 + poly0t;

    log_h = __CLC_USE_TABLE(powlog_tbl_head, index);
    log_t = __CLC_USE_TABLE(powlog_tbl_tail, index);

    __CLC_GENTYPE resT_t = __clc_fma(xexp, real_log2_tail, +log_t) - poly;
    __CLC_GENTYPE resT = resT_t - poly0h;
    __CLC_GENTYPE resH = __clc_fma(xexp, real_log2_lead, log_h);
    __CLC_GENTYPE resT_h = poly0h;

    __CLC_GENTYPE H = resT + resH;
    __CLC_GENTYPE H_h =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(H) & 0xfffffffff8000000L);
    __CLC_GENTYPE T =
        (resH - H + resT) + (resT_t - (resT + resT_h)) + (H - H_h);
    H = H_h;

    __CLC_GENTYPE y_head =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(uy) & 0xfffffffff8000000L);
    __CLC_GENTYPE y_tail = y - y_head;

    __CLC_GENTYPE fnyh =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(dny) & 0xfffffffffff00000);
    __CLC_GENTYPE fnyt = __CLC_CONVERT_GENTYPE(ny - __CLC_CONVERT_INTN(fnyh));
    y_tail = __clc_fma(-fnyt, y_head, __clc_fma(-fnyh, y_head, 1.0)) / dny;

    __CLC_GENTYPE temp = __clc_fma(y_tail, H, __clc_fma(y_head, T, y_tail * T));
    v = __clc_fma(y_head, H, temp);
    vt = __clc_fma(y_head, H, -v) + temp;
  }

  // Now calculate exp of (v,vt)

  __CLC_GENTYPE expv;
  {
    const __CLC_GENTYPE max_exp_arg = 709.782712893384;
    const __CLC_GENTYPE min_exp_arg = -745.1332191019411;
    const __CLC_GENTYPE sixtyfour_by_lnof2 = 92.33248261689366;
    const __CLC_GENTYPE lnof2_by_64_head = 0.010830424260348081;
    const __CLC_GENTYPE lnof2_by_64_tail = -4.359010638708991e-10;

    // If v is so large that we need to return INFINITY, or so small that we
    // need to return 0, set v to known values that will produce that result. Do
    // not try to continue the computation with the original v and patch it up
    // afterwards because v may be so large that temp is out of range of int, in
    // which case that conversion, and a value based on that conversion being
    // passed to __clc_ldexp, results in undefined behavior.
    v = v > max_exp_arg ? 1000.0 : v;
    v = v < min_exp_arg ? -1000.0 : v;

    __CLC_GENTYPE temp = v * sixtyfour_by_lnof2;
    __CLC_INTN n = __CLC_CONVERT_INTN(temp);
    __CLC_GENTYPE dn = __CLC_CONVERT_GENTYPE(n);
    __CLC_INTN j = n & 0x0000003f;
    __CLC_INTN m = n >> 6;

    __CLC_GENTYPE f1 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_head, j);
    __CLC_GENTYPE f2 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_tail, j);
    __CLC_GENTYPE f = f1 + f2;

    __CLC_GENTYPE r1 = __clc_fma(dn, -lnof2_by_64_head, v);
    __CLC_GENTYPE r2 = dn * lnof2_by_64_tail;
    __CLC_GENTYPE r = (r1 + r2) + vt;

    __CLC_GENTYPE q =
        __clc_fma(r,
                  __clc_fma(r,
                            __clc_fma(r,
                                      __clc_fma(r, 1.38889490863777199667e-03,
                                                8.33336798434219616221e-03),
                                      4.16666666662260795726e-02),
                            1.66666666665260878863e-01),
                  5.00000000000000008883e-01);
    q = __clc_fma(r * r, q, r);

    expv = __clc_fma(f, q, f2) + f1;
    expv = __clc_ldexp(expv, m);
  }

  // See whether y is an integer.
  // inty = 0 means not an integer.
  // inty = 1 means odd integer.
  // inty = 2 means even integer.

  __CLC_LONGN inty = __CLC_CONVERT_LONGN(2 - (ny & 1));

  expv *= ((inty == 1) & !xpos) ? -1.0 : 1.0;

  __CLC_LONGN ret = __CLC_AS_LONGN(expv);

  // Now all the edge cases
  __CLC_BIT_INTN x_is_ninf = ux == (__CLC_LONGN)NINFBITPATT_DP64;
  __CLC_BIT_INTN x_is_pinf = ux == (__CLC_LONGN)PINFBITPATT_DP64;
  ret = (!xpos & (inty == 2)) ? __CLC_AS_LONGN(__CLC_GENTYPE_NAN) : ret;
  __CLC_LONGN xinf =
      xpos ? (__CLC_LONGN)PINFBITPATT_DP64 : (__CLC_LONGN)NINFBITPATT_DP64;
  ret = ((ax == 0L) & !ypos & (inty == 1)) ? xinf : ret;
  ret =
      ((ax == 0L) & !ypos & (inty == 2)) ? (__CLC_LONGN)PINFBITPATT_DP64 : ret;
  ret = ((ax == 0L) & ypos & (inty == 2)) ? 0L : ret;
  __CLC_LONGN xzero = xpos ? 0L : (__CLC_LONGN)0x8000000000000000L;
  ret = ((ax == 0L) & ypos & (inty == 1)) ? xzero : ret;
  ret = (x_is_ninf & ypos & (inty == 1)) ? (__CLC_LONGN)NINFBITPATT_DP64 : ret;
  ret = (x_is_ninf & !ypos & (inty == 1)) ? (__CLC_LONGN)0x8000000000000000L
                                          : ret;
  ret = (x_is_pinf & !ypos) ? 0L : ret;
  ret = (x_is_pinf & ypos) ? (__CLC_LONGN)PINFBITPATT_DP64 : ret;
  ret = ax > (__CLC_LONGN)PINFBITPATT_DP64 ? ux : ret;
  ret = __CLC_CONVERT_LONGN(ny == 0) ? __CLC_AS_LONGN(__CLC_GENTYPE_NAN) : ret;
  return __CLC_AS_GENTYPE(ret);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_rootn(__CLC_GENTYPE x,
                                                 __CLC_INTN y) {
  return __CLC_CONVERT_GENTYPE(__clc_rootn(__CLC_CONVERT_FLOATN(x), y));
}

#endif
