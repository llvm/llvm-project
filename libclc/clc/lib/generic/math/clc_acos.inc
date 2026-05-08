//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes arccos(x).
//
// The incoming argument is first reduced by noting that arccos(x) is invalid
// for abs(x) > 1.
//
// For denormal and small arguments arccos(x) = pi/2 to machine accuracy.
//
// Remaining argument ranges are handled as follows:
// * For abs(x) <= 0.5 use:
//     arccos(x) = pi/2 - arcsin(x) = pi/2 - (x + x^3 * R(x^2))
//   where R(x^2) is a rational minimax approximation to (arcsin(x) - x)/x^3.
// * For abs(x) > 0.5 exploit the identity:
//     arccos(x) = pi - 2 * arcsin(sqrt(1 - x)/2)
//   together with the above rational approximation, and reconstruct the terms
//   carefully.
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32
_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_acos(__CLC_GENTYPE x) {
  // Computes arccos(x).
  // The argument is first reduced by noting that arccos(x)
  // is invalid for abs(x) > 1 and arccos(-x) = arccos(x).
  // For denormal and small arguments arccos(x) = pi/2 to machine
  // accuracy. Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arccos(x) = pi/2 - arcsin(x)
  // = pi/2 - (x + x^3*R(x^2))
  // where R(x^2) is a rational minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
  // together with the above rational approximation, and
  // reconstruct the terms carefully.

  __CLC_GENTYPE ax = __clc_fabs(x);

  __CLC_GENTYPE rt = __clc_mad(-0.5f, ax, 0.5f);
  __CLC_GENTYPE x2 = ax * ax;
  __CLC_GENTYPE r = ax > 0.5f ? rt : x2;

  __CLC_GENTYPE u =
      r * __clc_mad(r,
                    __clc_mad(r,
                              __clc_mad(r,
                                        __clc_mad(r,
                                                  __clc_mad(r, 0x1.38434ep-5f,
                                                            0x1.bf8bb4p-7f),
                                                  0x1.069878p-5f),
                                        0x1.6c8362p-5f),
                              0x1.33379p-4f),
                    0x1.555558p-3f);

  __CLC_GENTYPE s = __clc_sqrt_fast(r);
  __CLC_GENTYPE ztp = 2.0f * __clc_mad(s, u, s);
  __CLC_GENTYPE ztn = __clc_mad(0x1.ddcb02p+0f, 0x1.aee9d6p+0f, -ztp);
  __CLC_GENTYPE zt = x < 0.0f ? ztn : ztp;
  __CLC_GENTYPE z =
      __clc_mad(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -__clc_mad(x, u, x));
  z = ax > 0.5f ? zt : z;

  return z;
}

#elif __CLC_FPSIZE == 64

static _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_acos_identity_reduction(
    __CLC_GENTYPE x, __CLC_GENTYPE r, __CLC_GENTYPE u, __CLC_GENTYPE z) {
  __CLC_EP_PAIR s = __clc_ep_sqrt(r);
  __CLC_GENTYPE zm = __clc_mad(0x1.dd9ad336a0500p+0, 0x1.af154eeb562d6p+0,
                               -2.0 * __clc_mad(s.hi, u, s.hi));
  __CLC_GENTYPE zp = 2.0 * (s.hi + __clc_mad(s.hi, u, s.lo));
  z = x < 0.0 ? zm : zp;
  z = x == -1.0 ? 0x1.921fb54442d18p+1 : z;
  z = x == 1.0 ? 0.0 : z;
  return z;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_acos(__CLC_GENTYPE x) {
  // Computes arccos(x).
  // The argument is first reduced by noting that arccos(x)
  // is invalid for abs(x) > 1. For denormal and small
  // arguments arccos(x) = pi/2 to machine accuracy.
  // Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arccos(x) = pi/2 - arcsin(x)
  // = pi/2 - (x + x^3*R(x^2))
  // where R(x^2) is a rational minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
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

  __CLC_GENTYPE z = __clc_mad(0x1.dd9ad336a0500p-1, 0x1.af154eeb562d6p+0,
                              -__clc_mad(x, u, x));

#ifdef __CLC_SCALAR
  if (transform)
    z = __clc_acos_identity_reduction(x, r, u, z);
#else
  __CLC_GENTYPE identity = __clc_acos_identity_reduction(x, r, u, z);
  z = transform ? identity : z;
#endif

  return z;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE __clc_acos(__CLC_GENTYPE x) {
  // Computes arccos(x).
  // The argument is first reduced by noting that arccos(x)
  // is invalid for abs(x) > 1 and arccos(-x) = arccos(x).
  // For denormal and small arguments arccos(x) = pi/2 to machine
  // accuracy. Remaining argument ranges are handled as follows.
  // For abs(x) <= 0.5 use
  // arccos(x) = pi/2 - arcsin(x)
  // = pi/2 - (x + x^3*R(x^2))
  // where R(x^2) is a rational minimax approximation to
  // (arcsin(x) - x)/x^3.
  // For abs(x) > 0.5 exploit the identity:
  // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
  // together with the above rational approximation, and
  // reconstruct the terms carefully.

  __CLC_GENTYPE ax = __clc_fabs(x);

  __CLC_GENTYPE rt = __clc_mad(-0.5h, ax, 0.5h);
  __CLC_GENTYPE x2 = ax * ax;
  __CLC_GENTYPE r = ax > 0.5h ? rt : x2;

  __CLC_GENTYPE u = r * __clc_mad(r, 0x1.828p-4h, 0x1.52p-3h);

  __CLC_GENTYPE s = __clc_sqrt_fast(r);
  __CLC_GENTYPE ztp = 2.0h * __clc_mad(s, u, s);
  __CLC_GENTYPE ztn = __clc_mad(0x1.ea8p+0h, 0x1.a3cp+0h, -ztp);
  __CLC_GENTYPE zt = x < 0.0h ? ztn : ztp;
  __CLC_GENTYPE z = __clc_mad(0x1.ea8p-1h, 0x1.a3cp+0h, -__clc_mad(x, u, x));
  z = ax > 0.5h ? zt : z;

  return z;
}

#endif
