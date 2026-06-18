//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#if __CLC_FPSIZE != 16
/// \returns true if the fractional part of \p x is 0. This is equivalent to
// __clc_fract(x, / unused) == 0.0
static _CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_fract_is_zero(__CLC_GENTYPE x) {
  return __clc_trunc(x) == x;
}
#endif

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_tgamma(__CLC_FLOATN x) {
  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_FLOATN ret;

  if (ax < 16.0f) {
    __CLC_FLOATN n, d;
    __CLC_FLOATN y = x;
    if (x > 0.0f) {
      n = 1.0f;

      while (y > 2.5f) {
        n = __clc_mad(n, y, -n);
        y = y - 1.0f;
        n = __clc_mad(n, y, -n);
        y = y - 1.0f;
      }

      if (y > 1.5f) {
        n = __clc_mad(n, y, -n);
        y = y - 1.0f;
      }

      if (x >= 0.5f)
        y = y - 1.0f;

      d = x < 0.5f ? x : 1.0f;
    } else {
      d = x;

      while (y < -1.5f) {
        d = __clc_mad(d, y, d);
        y = y + 1.0f;
        d = __clc_mad(d, y, d);
        y = y + 1.0f;
      }

      if (y < -0.5f) {
        d = __clc_mad(d, y, d);
        y = y + 1.0f;
      }

      n = 1.0f;
    }

    __CLC_FLOATN p0 = __clc_mad(y, 0x1.d5a56ep-8f, -0x1.4dcb00p-7f);
    __CLC_FLOATN p1 = __clc_mad(y, p0, -0x1.59c03ap-5f);
    __CLC_FLOATN p2 = __clc_mad(y, p1, 0x1.55405ap-3f);
    __CLC_FLOATN p3 = __clc_mad(y, p2, -0x1.5810f2p-5f);
    __CLC_FLOATN p4 = __clc_mad(y, p3, -0x1.4fcfd6p-1f);
    __CLC_FLOATN qt = __clc_mad(y, p4, 0x1.2788ccp-1f);

    ret = n / __clc_mad(d, y * qt, d);
    ret = x == 0.0f ? __clc_copysign(__CLC_GENTYPE_INF, x) : ret;
    ret = (x < 0.0f && __clc_fract_is_zero(x)) ? FLT_NAN : ret;
  } else {
    const __CLC_FLOATN sqrt2pi = 0x1.40d932p+1f;
    const __CLC_FLOATN sqrtpiby2 = 0x1.40d932p+0f;

    __CLC_FLOATN t1 = __clc_powr(ax, __clc_mad(ax, 0.5f, -0.25f));
    __CLC_FLOATN t2 = __clc_exp(-ax);
    __CLC_FLOATN xr = __clc_recip_fast(ax);
    __CLC_FLOATN p = __clc_mad(
        xr, __clc_mad(xr, 0x1.96d7e4p-9f, 0x1.556652p-4f), 0x1.fffff8p-1f);
    if (x > 0.0f) {
      __CLC_FLOATN g = sqrt2pi * t2 * t1 * t1 * p;
      ret = x > 0x1.18521ep+5f ? __CLC_GENTYPE_INF : g;
    } else {
      __CLC_FLOATN s = -x * __clc_sinpi(x);
      if (x > -30.0f)
        ret = sqrtpiby2 / (s * t2 * t1 * t1 * p);
      else if (x > -41.0f)
        ret = (sqrtpiby2 / (t2 * t1 * p)) / (s * t1);
      else
        ret = __clc_copysign(0.0f, s);

      ret = __clc_fract_is_zero(x) || __clc_isnan(x) ? FLT_NAN : ret;
    }
  }

  return ret;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_tgamma(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN ax = __clc_fabs(x);
  __CLC_DOUBLEN ret;

  if (ax < 16.0) {
    __CLC_DOUBLEN n, d;
    __CLC_DOUBLEN y = x;
    if (x > 0.0) {
      n = 1.0;
      while (y > 2.5) {
        n = __clc_mad(n, y, -n);
        y = y - 1.0;
        n = __clc_mad(n, y, -n);
        y = y - 1.0;
      }

      if (y > 1.5) {
        n = __clc_mad(n, y, -n);
        y = y - 1.0;
      }

      if (x >= 0.5)
        y = y - 1.0;

      d = x < 0.5 ? x : 1.0;
    } else {
      d = x;

      while (y < -1.5) {
        d = __clc_mad(d, y, d);
        y = y + 1.0;
        d = __clc_mad(d, y, d);
        y = y + 1.0;
      }

      if (y < -0.5) {
        d = __clc_mad(d, y, d);
        y = y + 1.0;
      }

      n = 1.0;
    }

    __CLC_DOUBLEN t0 =
        __clc_mad(y, -0x1.aed75feec7b9ap-23, 0x1.31854a0be3cd3p-20);
    __CLC_DOUBLEN t1 = __clc_mad(y, t0, -0x1.5037d6a97a8b7p-20);
    __CLC_DOUBLEN t2 = __clc_mad(y, t1, -0x1.51d67f2cdbcfbp-16);
    __CLC_DOUBLEN t3 = __clc_mad(y, t2, 0x1.0c8ab2ac5112dp-13);
    __CLC_DOUBLEN t4 = __clc_mad(y, t3, -0x1.c364ce9b5e149p-13);
    __CLC_DOUBLEN t5 = __clc_mad(y, t4, -0x1.317113a39f929p-10);
    __CLC_DOUBLEN t6 = __clc_mad(y, t5, 0x1.d919c501178a3p-8);
    __CLC_DOUBLEN t7 = __clc_mad(y, t6, -0x1.3b4af282da690p-7);
    __CLC_DOUBLEN t8 = __clc_mad(y, t7, -0x1.59af103bf2cd0p-5);
    __CLC_DOUBLEN t9 = __clc_mad(y, t8, 0x1.5512320b432ccp-3);
    __CLC_DOUBLEN t10 = __clc_mad(y, t9, -0x1.5815e8fa28886p-5);
    __CLC_DOUBLEN t11 = __clc_mad(y, t10, -0x1.4fcf4026afa24p-1);
    __CLC_DOUBLEN qt = __clc_mad(y, t11, 0x1.2788cfc6fb61cp-1);

    ret = n / __clc_mad(d, y * qt, d);
    ret = x == 0.0 ? __clc_copysign(__CLC_GENTYPE_INF, x) : ret;
    ret = (x < 0.0 && __clc_fract_is_zero(x)) ? DBL_NAN : ret;
  } else {
    const __CLC_DOUBLEN sqrt2pi = 0x1.40d931ff62706p+1;
    const __CLC_DOUBLEN sqrtpiby2 = 0x1.40d931ff62706p+0;

    __CLC_DOUBLEN t1 = __clc_powr(ax, __clc_mad(ax, 0.5, -0.25));
    __CLC_DOUBLEN t2 = __clc_exp(-ax);
    __CLC_DOUBLEN xr = __clc_recip_fast(ax);

    __CLC_DOUBLEN p0 =
        __clc_mad(xr, -0x1.2b04c5ea74bbfp-11, 0x1.14869344f1d9bp-14);
    __CLC_DOUBLEN p1 = __clc_mad(xr, p0, 0x1.9b3457156ffefp-11);
    __CLC_DOUBLEN p2 = __clc_mad(xr, p1, -0x1.e1427e86ee097p-13);
    __CLC_DOUBLEN p3 = __clc_mad(xr, p2, -0x1.5f7266f67c4e0p-9);
    __CLC_DOUBLEN p4 = __clc_mad(xr, p3, 0x1.c71c71c0f96adp-9);
    __CLC_DOUBLEN pt = __clc_mad(xr, p4, 0x1.5555555555a28p-4);

    if (x > 0.0) {
      __CLC_DOUBLEN gt = sqrt2pi * t2 * t1 * t1;
      __CLC_DOUBLEN g = __clc_mad(gt, xr * pt, gt);
      ret = x > 0x1.573fae561f646p+7 ? __CLC_GENTYPE_INF : g;
    } else {
      __CLC_DOUBLEN s = -x * __clc_sinpi(x);
      if (x > -170.5) {
        __CLC_DOUBLEN d = s * t2 * t1 * t1;
        ret = sqrtpiby2 / __clc_mad(d, xr * pt, d);
      } else if (x > -184.0) {
        __CLC_DOUBLEN d = t2 * t1;
        ret = (sqrtpiby2 / __clc_mad(d, xr * pt, d)) / (s * t1);
      } else
        ret = __clc_copysign(0.0, s);

      ret = __clc_fract_is_zero(x) || __clc_isnan(x) ? DBL_NAN : ret;
    }
  }

  return ret;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_tgamma(__CLC_HALFN x) {
  return __CLC_CONVERT_HALFN(__clc_tgamma(__CLC_CONVERT_FLOATN(x)));
}

#endif

#endif // __CLC_SCALAR
