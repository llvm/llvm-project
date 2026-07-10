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
//    where R(x^2) is a rational minimax approximation to (arcsin(x) - x)/x^3.
// * For abs(x) > 0.5 exploit the identity:
//     arccos(x) = pi - 2 * arcsin(sqrt(1 - x)/2)
//   together with the above rational approximation, and reconstruct the terms
//   carefully.
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_acospi(__CLC_FLOATN x) {
  const __CLC_FLOATN piinv = 0x1.45f306p-2f;

  __CLC_FLOATN ax = __clc_fabs(x);

  __CLC_FLOATN rt = __clc_mad(-0.5f, ax, 0.5f);
  __CLC_FLOATN x2 = ax * ax;
  __CLC_FLOATN r = ax > 0.5f ? rt : x2;

  __CLC_FLOATN u = r * __clc_mad(r, __clc_mad(r, __clc_mad(r, __clc_mad(r,
                __clc_mad(r,
                    -0x1.3f1c6cp-8f, 0x1.2ac560p-6f), 0x1.80aab4p-8f), 0x1.e53378p-7f),
                    0x1.86680ap-6f), 0x1.b29c5ap-5f);

  __CLC_FLOATN s = __clc_sqrt_fast(r);
  __CLC_FLOATN ztp = 2.0f * __clc_mad(s, u, piinv * s);
  __CLC_FLOATN ztn = 1.0f - ztp;
  __CLC_FLOATN zt = x < 0.0f ? ztn : ztp;
  __CLC_FLOATN z = 0.5f - __clc_mad(x, u, piinv * x);
  return ax > 0.5f ? zt : z;
}

#elif __CLC_FPSIZE == 64

#define piinv 0x1.45f306dc9c883p-2

static _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_acospi_identity(
    __CLC_DOUBLEN x, __CLC_DOUBLEN r, __CLC_DOUBLEN u, __CLC_DOUBLEN z) {
  __CLC_EP_PAIR s = __clc_ep_ldexp(__clc_ep_sqrt(r), 1);
  __CLC_DOUBLEN zm = 1.0 - __clc_mad(s.hi, u, piinv * s.hi);
  __CLC_EP_PAIR zp =
      __clc_ep_fast_add(__clc_ep_mul(piinv, s), __clc_ep_mul(s, u));
  z = x < 0.0 ? zm : zp.hi;
  z = x == -1.0 ? 1.0 : z;
  return x == 1.0 ? 0.0 : z;
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_acospi(__CLC_DOUBLEN x) {
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

  __CLC_DOUBLEN y = __clc_fabs(x);
  __CLC_LONGN transform = y >= 0.5;

  __CLC_DOUBLEN rt = __clc_mad(y, -0.5, 0.5);
  __CLC_DOUBLEN y2 = y * y;
  __CLC_DOUBLEN r = transform ? rt : y2;

  __CLC_DOUBLEN u = r * __clc_mad(r, __clc_mad(r, __clc_mad(r, __clc_mad(r,
                 __clc_mad(r, __clc_mad(r, __clc_mad(r, __clc_mad(r,
                 __clc_mad(r, __clc_mad(r, __clc_mad(r,
                     0x1.547a51d41fb0bp-7, -0x1.6a3fb0718a8f7p-8), 0x1.a7b91f7177ee8p-8), 0x1.035d3435b8ad8p-9),
                     0x1.ff0549b4e0449p-9), 0x1.21604ae288f96p-8), 0x1.6a2b36f9aec49p-8), 0x1.d2b076c914f04p-8),
                     0x1.3ce53861f8f1fp-7), 0x1.d1a4529a30a69p-7), 0x1.8723a1d61d2e9p-6), 0x1.b2995e7b7af0fp-5);

  __CLC_DOUBLEN z = 0.5 - __clc_mad(x, u, piinv * x);

  return transform ? __clc_acospi_identity(x, r, u, z) : z;
}

#undef piinv

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_acospi(__CLC_HALFN x) {
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

  const __CLC_HALFN piinv = 0x1.46p-2h;

  __CLC_HALFN ax = __clc_fabs(x);

  __CLC_HALFN rt = __clc_mad(-0.5h, ax, 0.5h);
  __CLC_HALFN x2 = ax * ax;
  __CLC_HALFN r = ax > 0.5h ? rt : x2;

  __CLC_HALFN u = r * __clc_mad(r, 0x1.0b8p-5h, 0x1.a7cp-5h);

  __CLC_HALFN s = __clc_sqrt_fast(r);
  __CLC_HALFN ztp = 2.0h * __clc_mad(s, u, piinv * s);
  __CLC_HALFN ztn = 1.0h - ztp;
  __CLC_HALFN zt = x < 0.0h ? ztn : ztp;
  __CLC_HALFN z = 0.5h - __clc_mad(x, u, piinv * x);
  return ax > 0.5h ? zt : z;
}

#endif
