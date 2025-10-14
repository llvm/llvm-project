//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atan2(__CLC_GENTYPE y,
                                                 __CLC_GENTYPE x) {
  const __CLC_GENTYPE pi = 0x1.921fb6p+1f;
  const __CLC_GENTYPE piby2 = 0x1.921fb6p+0f;
  const __CLC_GENTYPE piby4 = 0x1.921fb6p-1f;
  const __CLC_GENTYPE threepiby4 = 0x1.2d97c8p+1f;

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE ay = __clc_fabs(y);
  __CLC_GENTYPE v = __clc_min(ax, ay);
  __CLC_GENTYPE u = __clc_max(ax, ay);

  // Scale since u could be large, as in "regular" divide
  __CLC_GENTYPE s = u > 0x1.0p+96f ? 0x1.0p-32f : 1.0f;
  __CLC_GENTYPE vbyu = s * MATH_DIVIDE(v, s * u);

  __CLC_GENTYPE vbyu2 = vbyu * vbyu;

#define USE_2_2_APPROXIMATION
#if defined USE_2_2_APPROXIMATION
  __CLC_GENTYPE p =
      __clc_mad(vbyu2, __clc_mad(vbyu2, -0x1.7e1f78p-9f, -0x1.7d1b98p-3f),
                -0x1.5554d0p-2f) *
      vbyu2 * vbyu;
  __CLC_GENTYPE q =
      __clc_mad(vbyu2, __clc_mad(vbyu2, 0x1.1a714cp-2f, 0x1.287c56p+0f), 1.0f);
#else
  __CLC_GENTYPE p =
      __clc_mad(vbyu2, __clc_mad(vbyu2, -0x1.55cd22p-5f, -0x1.26cf76p-2f),
                -0x1.55554ep-2f) *
      vbyu2 * vbyu;
  __CLC_GENTYPE q = __clc_mad(
      vbyu2,
      __clc_mad(vbyu2, __clc_mad(vbyu2, 0x1.9f1304p-5f, 0x1.2656fap-1f),
                0x1.76b4b8p+0f),
      1.0f);
#endif

  // Octant 0 result
  __CLC_GENTYPE a = __clc_mad(p, MATH_RECIP(q), vbyu);

  // Fix up 3 other octants
  __CLC_GENTYPE at = piby2 - a;
  a = ay > ax ? at : a;
  at = pi - a;
  a = x < 0.0F ? at : a;

  // y == 0 => 0 for x >= 0, pi for x < 0
  at = __CLC_AS_INTN(x) < 0 ? pi : 0.0f;
  a = y == 0.0f ? at : a;

  // x and y are +- Inf
  at = x > 0.0f ? piby4 : threepiby4;
  a = __clc_select(a, at, __clc_isinf(x) && __clc_isinf(y));

  // x or y is NaN
  a = __clc_select(a, __CLC_GENTYPE_NAN, __clc_isnan(x) || __clc_isnan(y));

  // Fixup sign and return
  return __clc_copysign(a, y);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atan2(__CLC_GENTYPE y,
                                                 __CLC_GENTYPE x) {
  const __CLC_GENTYPE pi = 3.1415926535897932e+00;    /* 0x400921fb54442d18 */
  const __CLC_GENTYPE piby2 = 1.5707963267948966e+00; /* 0x3ff921fb54442d18 */
  const __CLC_GENTYPE piby4 = 7.8539816339744831e-01; /* 0x3fe921fb54442d18 */
  // 0x4002d97c7f3321d2
  const __CLC_GENTYPE three_piby4 = 2.3561944901923449e+00;
  const __CLC_GENTYPE pi_head = 3.1415926218032836e+00; /* 0x400921fb50000000 */
  const __CLC_GENTYPE pi_tail = 3.1786509547056392e-08; /* 0x3e6110b4611a6263 */
  // 0x3ff921fb54442d18
  const __CLC_GENTYPE piby2_head = 1.5707963267948965e+00;
  // 0x3c91a62633145c07
  const __CLC_GENTYPE piby2_tail = 6.1232339957367660e-17;

  __CLC_GENTYPE x2 = x;
  // Important to capture -0.0 in xneg and yneg, so comparison done as integer
  __CLC_LONGN xneg = __CLC_AS_LONGN(x) < 0;
  __CLC_INTN xexp =
      __CLC_CONVERT_INTN(__CLC_AS_ULONGN(x) >> EXPSHIFTBITS_DP64) & 0x7ff;

  __CLC_GENTYPE y2 = y;
  __CLC_LONGN yneg = __CLC_AS_LONGN(y) < 0;
  __CLC_INTN yexp =
      __CLC_CONVERT_INTN(__CLC_AS_ULONGN(y) >> EXPSHIFTBITS_DP64) & 0x7ff;

  __CLC_LONGN cond2 = __CLC_CONVERT_LONGN(xexp < 1021 && yexp < 1021);
  __CLC_LONGN diffexp = __CLC_CONVERT_LONGN(yexp - xexp);

  // Scale up both x and y if they are both below 1/4
  __CLC_GENTYPE x1 = __clc_ldexp(x, 1024);
  __CLC_INTN xexp1 =
      __CLC_CONVERT_INTN(__CLC_AS_ULONGN(x1) >> EXPSHIFTBITS_DP64) & 0x7ff;
  __CLC_GENTYPE y1 = __clc_ldexp(y, 1024);
  __CLC_INTN yexp1 =
      __CLC_CONVERT_INTN(__CLC_AS_ULONGN(y1) >> EXPSHIFTBITS_DP64) & 0x7ff;
  __CLC_LONGN diffexp1 = __CLC_CONVERT_LONGN(yexp1 - xexp1);

  diffexp = __clc_select(diffexp, diffexp1, cond2);
  x = cond2 ? x1 : x;
  y = cond2 ? y1 : y;

  // General case: take absolute values of arguments
  __CLC_GENTYPE u = __clc_fabs(x);
  __CLC_GENTYPE v = __clc_fabs(y);

  // Swap u and v if necessary to obtain 0 < v < u. Compute v/u.
  __CLC_LONGN swap_vu = u < v;
  __CLC_GENTYPE uu = u;
  u = swap_vu ? v : u;
  v = swap_vu ? uu : v;

  __CLC_GENTYPE vbyu = v / u;
  __CLC_GENTYPE q1, q2;

  // General values of v/u. Use a look-up table and series expansion.

  {
    __CLC_GENTYPE val = vbyu > 0.0625 ? vbyu : 0.063;
    __CLC_INTN index = __CLC_CONVERT_INTN(__clc_fma(256.0, val, 0.5));
    q1 = __CLC_USE_TABLE(atan_jby256_tbl_head, index - 16);
    q2 = __CLC_USE_TABLE(atan_jby256_tbl_tail, index - 16);
    __CLC_GENTYPE c = __CLC_CONVERT_GENTYPE(index) * 0x1.0p-8;

    // We're going to scale u and v by 2^(-u_exponent) to bring them close to 1
    // u_exponent could be EMAX so we have to do it in 2 steps
    __CLC_INTN m =
        -(__CLC_CONVERT_INTN(__CLC_AS_ULONGN(u) >> EXPSHIFTBITS_DP64) -
          EXPBIAS_DP64);
    __CLC_GENTYPE um = __clc_ldexp(u, m);
    __CLC_GENTYPE vm = __clc_ldexp(v, m);

    // 26 leading bits of u
    __CLC_GENTYPE u1 =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(um) & 0xfffffffff8000000UL);
    __CLC_GENTYPE u2 = um - u1;

    __CLC_GENTYPE r = MATH_DIVIDE(__clc_fma(-c, u2, __clc_fma(-c, u1, vm)),
                                  __clc_fma(c, vm, um));

    // Polynomial approximation to atan(r)
    __CLC_GENTYPE s = r * r;
    q2 = q2 + __clc_fma((s * __clc_fma(-s, 0.19999918038989143496,
                                       0.33333333333224095522)),
                        -r, r);
  }

  __CLC_GENTYPE q3, q4;
  {
    q3 = 0.0;
    q4 = vbyu;
  }

  __CLC_GENTYPE q5, q6;
  {
    __CLC_GENTYPE u1 =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(u) & 0xffffffff00000000UL);
    __CLC_GENTYPE u2 = u - u1;
    __CLC_GENTYPE vu1 =
        __CLC_AS_GENTYPE(__CLC_AS_ULONGN(vbyu) & 0xffffffff00000000UL);
    __CLC_GENTYPE vu2 = vbyu - vu1;

    q5 = 0.0;
    __CLC_GENTYPE s = vbyu * vbyu;
    q6 = vbyu +
         __clc_fma(
             -vbyu * s,
             __clc_fma(
                 -s,
                 __clc_fma(-s,
                           __clc_fma(-s,
                                     __clc_fma(-s, 0.90029810285449784439E-01,
                                               0.11110736283514525407),
                                     0.14285713561807169030),
                           0.19999999999393223405),
                 0.33333333333333170500),
             MATH_DIVIDE(__clc_fma(-u, vu2,
                                   __clc_fma(-u2, vu1, __clc_fma(-u1, vu1, v))),
                         u));
  }

  q3 = vbyu < 0x1.d12ed0af1a27fp-27 ? q3 : q5;
  q4 = vbyu < 0x1.d12ed0af1a27fp-27 ? q4 : q6;

  q1 = vbyu > 0.0625 ? q1 : q3;
  q2 = vbyu > 0.0625 ? q2 : q4;

  // Tidy-up according to which quadrant the arguments lie in
  __CLC_GENTYPE res1, res2, res3, res4;
  q1 = swap_vu ? piby2_head - q1 : q1;
  q2 = swap_vu ? piby2_tail - q2 : q2;
  q1 = xneg ? pi_head - q1 : q1;
  q2 = xneg ? pi_tail - q2 : q2;
  q1 = q1 + q2;
  res4 = yneg ? -q1 : q1;

  res1 = yneg ? -three_piby4 : three_piby4;
  res2 = yneg ? -piby4 : piby4;
  res3 = xneg ? res1 : res2;

  res3 = __clc_select(res4, res3,
                      __CLC_CONVERT_LONGN(__clc_isinf(x2) && __clc_isinf(y2)));
  res1 = yneg ? -pi : pi;

  // abs(x)/abs(y) > 2^56 and x < 0
  res3 = (diffexp < -56 && xneg) ? res1 : res3;

  res4 = MATH_DIVIDE(y, x);
  // x positive and dominant over y by a factor of 2^28
  res3 = diffexp < -28 && xneg == 0 ? res4 : res3;

  // abs(y)/abs(x) > 2^56
  res4 = yneg ? -piby2 : piby2; // atan(y/x) is insignificant compared to piby2
  res3 = diffexp > 56 ? res4 : res3;

  res3 = x2 == 0.0 ? res4 : res3; // Zero x gives +- pi/2 depending on sign of y
  res4 = xneg ? res1 : y2;

  // Zero y gives +-0 for positive x and +-pi for negative x
  res3 = y2 == 0.0 ? res4 : res3;
  res3 = __clc_isnan(y2) ? y2 : res3;
  res3 = __clc_isnan(x2) ? x2 : res3;

  return res3;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atan2(__CLC_GENTYPE x,
                                                 __CLC_GENTYPE y) {
  return __CLC_CONVERT_GENTYPE(
      __clc_atan2(__CLC_CONVERT_FLOATN(x), __CLC_CONVERT_FLOATN(y)));
}

#endif
