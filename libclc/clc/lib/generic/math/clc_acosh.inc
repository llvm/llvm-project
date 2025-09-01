//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_acosh(__CLC_GENTYPE x) {
  __CLC_UINTN ux = __CLC_AS_UINTN(x);

  // Arguments greater than 1/sqrt(epsilon) in magnitude are approximated by
  // acosh(x) = ln(2) + ln(x)
  // For 2.0 <= x <= 1/sqrt(epsilon) the approximation is:
  //   acosh(x) = ln(x + sqrt(x * x - 1))
  __CLC_INTN high = ux > 0x46000000U;
  __CLC_INTN med = ux > 0x40000000U;

  __CLC_GENTYPE w = x - 1.0f;
  __CLC_GENTYPE s = w * w + 2.0f * w;
  __CLC_GENTYPE t = x * x - 1.0f;
  __CLC_GENTYPE r = __clc_sqrt(med ? t : s) + (med ? x : w);
  __CLC_GENTYPE v = (high ? x : r) - (med ? 1.0f : 0.0f);
  __CLC_GENTYPE z = __clc_log1p(v) + (high ? 0x1.62e430p-1f : 0.0f);

  z = __clc_select(z, x, __clc_isnan(x) || __clc_isinf(x));
  z = x < 1.0f ? __CLC_GENTYPE_NAN : z;

  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_acosh(__CLC_GENTYPE x) {
  // 1/sqrt(eps) = 9.49062656242515593767e+07
  const __CLC_GENTYPE recrteps = 0x1.6a09e667f3bcdp+26;
  // log2_lead and log2_tail sum to an extra-precise version of log(2)
  const __CLC_GENTYPE log2_lead = 0x1.62e42ep-1;
  const __CLC_GENTYPE log2_tail = 0x1.efa39ef35793cp-25;

  // Handle x >= 128 here
  __CLC_LONGN xlarge = x > recrteps;
  __CLC_GENTYPE r = x + __clc_sqrt(__clc_fma(x, x, -1.0));
  r = xlarge ? x : r;

  __CLC_INTN xexp;
  __CLC_GENTYPE r1, r2;
  __clc_ep_log(r, &xexp, &r1, &r2);

  __CLC_GENTYPE dxexp = __CLC_CONVERT_GENTYPE(
      __CLC_CONVERT_LONGN(xexp) + (xlarge ? (__CLC_LONGN)1 : (__CLC_LONGN)0));
  r1 = __clc_fma(dxexp, log2_lead, r1);
  r2 = __clc_fma(dxexp, log2_tail, r2);

  __CLC_GENTYPE ret1 = r1 + r2;

  // Handle 1 < x < 128 here
  // We compute the value
  // t = x - 1.0 + sqrt(2.0*(x - 1.0) + (x - 1.0)*(x - 1.0))
  // using simulated quad precision.
  __CLC_GENTYPE t = x - 1.0;
  __CLC_GENTYPE u1 = t * 2.0;

  // (t,0) * (t,0) -> (v1, v2)
  __CLC_GENTYPE v1 = t * t;
  __CLC_GENTYPE v2 = __clc_fma(t, t, -v1);

  // (u1,0) + (v1,v2) -> (w1,w2)
  r = u1 + v1;
  __CLC_GENTYPE s = (((u1 - r) + v1) + v2);
  __CLC_GENTYPE w1 = r + s;
  __CLC_GENTYPE w2 = (r - w1) + s;

  // sqrt(w1,w2) -> (u1,u2)
  __CLC_GENTYPE p1 = __clc_sqrt(w1);
  __CLC_GENTYPE a1 = p1 * p1;
  __CLC_GENTYPE a2 = __clc_fma(p1, p1, -a1);
  __CLC_GENTYPE temp = (((w1 - a1) - a2) + w2);
  __CLC_GENTYPE p2 = MATH_DIVIDE(temp * 0.5, p1);
  u1 = p1 + p2;
  __CLC_GENTYPE u2 = (p1 - u1) + p2;

  // (u1,u2) + (t,0) -> (r1,r2)
  r = u1 + t;
  s = ((u1 - r) + t) + u2;
  // r1 = r + s;
  // r2 = (r - r1) + s;
  // t = r1 + r2;
  t = r + s;

  // For arguments 1.13 <= x <= 1.5 the log1p function is good enough
  __CLC_GENTYPE ret2 = __clc_log1p(t);

  __CLC_GENTYPE ret = x >= 128.0 ? ret1 : ret2;

  ret = (__clc_isinf(x) || __clc_isnan(x)) ? x : ret;
  ret = x == 1.0 ? 0.0 : ret;
  ret = x < 1.0 ? __CLC_GENTYPE_NAN : ret;

  return ret;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_acosh(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_acosh(__CLC_CONVERT_FLOATN(x)));
}

#endif
