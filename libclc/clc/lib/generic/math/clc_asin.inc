//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes arcsin(x).
//
// The incoming argument is first reduced by noting that arcsin(x) is invalid
// for abs(x) > 1 and arcsin(-x) = -arcsin(x).
//
// For denormal and small arguments, arcsin(x) = x to machine accuracy.
//
// Remaining argument ranges are handled as follows:
// * For abs(x) <= 0.5 use:
//     arcsin(x) = x + x^3 * R(x^2)
//   where R(x^2) is a rational minimax approximation to (arcsin(x) - x)/x^3.
// * For abs(x) > 0.5 exploit the identity:
//     arcsin(x) = pi/2 - 2 * arcsin(sqrt(1 - x)/2)
//   together with the above rational approximation, and reconstruct the terms
//   carefully.
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_asin(__CLC_GENTYPE x) {
  // Computes arcsin(x).
  // The argument is first reduced by noting that arcsin(x)
  // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
  // For denormal and small arguments arcsin(x) = x to machine
  // accuracy. Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arcsin(x) = x + x^3*R(x^2)
  // where R(x^2) is a polynomial minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
  // together with the above polynomial approximation, and
  // reconstruct the terms carefully.

  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE tx = __clc_mad(ax, -0.5f, 0.5f);
  __CLC_GENTYPE x2 = x * x;
  __CLC_GENTYPE r = ax >= 0.5f ? tx : x2;

  __CLC_GENTYPE u = r * __clc_mad(r, __clc_mad(r, __clc_mad(r, __clc_mad(r,
                __clc_mad(r,
                    0x1.38434ep-5f, 0x1.bf8bb4p-7f), 0x1.069878p-5f), 0x1.6c8362p-5f),
                    0x1.33379p-4f), 0x1.555558p-3f);

  __CLC_GENTYPE s = __clc_sqrt_fast(r);
  __CLC_GENTYPE ret =
      __clc_mad(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -2.0f * __clc_mad(s, u, s));

  __CLC_GENTYPE xux = __clc_mad(ax, u, ax);
  ret = ax < 0.5f ? xux : ret;

  return __clc_copysign(ret, x);
}

#elif __CLC_FPSIZE == 64

static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_asin_identity_reduction(__CLC_GENTYPE x, __CLC_GENTYPE r,
                                                                            __CLC_GENTYPE u, __CLC_GENTYPE v) {
  __CLC_GENTYPE y = __clc_fabs(x);
  __CLC_EP_PAIR s = __clc_ep_sqrt(r);
  __CLC_EP_PAIR ve = __clc_ep_fast_sub(
    __clc_ep_make_pair(__CLC_FP_LIT(0x1.921fb54442d18p-1), __CLC_FP_LIT(0x1.1a62633145c07p-55)),
    __clc_ep_fast_add(s, __clc_ep_mul(s, u)));
  v = ve.hi + ve.hi;
  return y == 1.0 ? 0x1.921fb54442d18p+0 : v;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_asin(__CLC_GENTYPE x) {
  // Computes arcsin(x).
  // The argument is first reduced by noting that arcsin(x)
  // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
  // For denormal and small arguments arcsin(x) = x to machine
  // accuracy. Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arcsin(x) = x + x^3*R(x^2)
  // where R(x^2) is a rational minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
  // together with the above rational approximation, and
  // reconstruct the terms carefully.

  __CLC_GENTYPE y = __clc_fabs(x);
  __CLC_S_GENTYPE transform = y >= 0.5;

  __CLC_GENTYPE rt = __clc_mad(y, -0.5, 0.5);
  __CLC_GENTYPE y2 = y * y;
  __CLC_GENTYPE r = transform ? rt : y2;

  __CLC_GENTYPE u = r * __clc_mad(r, __clc_mad(r, __clc_mad(r, __clc_mad(r,
                 __clc_mad(r, __clc_mad(r, __clc_mad(r, __clc_mad(r,
                 __clc_mad(r, __clc_mad(r, __clc_mad(r,
                     0x1.059859fea6a70p-5, -0x1.0a5a378a05eafp-6), 0x1.4052137024d6ap-6), 0x1.ab3a098a70509p-8),
                     0x1.8ed60a300c8d2p-7), 0x1.c6fa84b77012bp-7), 0x1.1c6c111dccb70p-6), 0x1.6e89f0a0adacfp-6),
                     0x1.f1c72c668963fp-6), 0x1.6db6db41ce4bdp-5), 0x1.333333336fd5bp-4), 0x1.5555555555380p-3);

  __CLC_GENTYPE v = __clc_mad(y, u, y);

#ifdef __CLC_SCALAR
  if (transform) {
    v = __clc_asin_identity_reduction(x, r, u, v);
  }
#else
  __CLC_GENTYPE identity = __clc_asin_identity_reduction(x, r, u, v);
  v = transform ? identity : v;
#endif

  return __clc_copysign(v, x);
}

#elif __CLC_FPSIZE == 16

static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_asin_small_case(__CLC_GENTYPE x) {
  __CLC_HALFN ax = __clc_fabs(x);
  __CLC_HALFN s = x * x;
  __CLC_HALFN p = s * __clc_mad(s, 0x1.828p-4h, 0x1.52p-3h);
  return __clc_mad(ax, p, ax);
}

static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_asin_large_case(__CLC_GENTYPE x) {
  __CLC_HALFN ax = __clc_fabs(x);
  __CLC_FLOATN s = __clc_mad(__CLC_CONVERT_FLOATN(ax), (__CLC_FLOATN)-0.5f,
                             (__CLC_FLOATN)0.5f);
  __CLC_FLOATN t = __clc_sqrt_fast(s);
  __CLC_FLOATN p = __clc_mad(t, __clc_mad(s, -0x1.82675ap-2f, -0x1.ff9f6p+0f),
                             0x1.921fb6p+0f);
  return __CLC_CONVERT_HALFN(p);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_asin(__CLC_GENTYPE x) {
  // Computes arcsin(x).
  // The argument is first reduced by noting that arcsin(x)
  // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
  // For denormal and small arguments arcsin(x) = x to machine
  // accuracy. Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arcsin(x) = x + x^3*R(x^2)
  // where R(x^2) is a polynomial minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
  // together with the above polynomial approximation, and
  // reconstruct the terms carefully.

  __CLC_HALFN ax = __clc_fabs(x);
  __CLC_HALFN r =
      ax <= 0.5h ? __clc_asin_small_case(x) : __clc_asin_large_case(x);
  return __clc_copysign(r, x);
}

#endif
