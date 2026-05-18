//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma OPENCL FP_CONTRACT OFF

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_FLOATN __clc_atan2(__CLC_FLOATN y,
                                                           __CLC_FLOATN x) {
  const __CLC_FLOATN pi = 0x1.921fb6p+1f;
  const __CLC_FLOATN piby2 = 0x1.921fb6p+0f;
  const __CLC_FLOATN piby4 = 0x1.921fb6p-1f;
  const __CLC_FLOATN threepiby4 = 0x1.2d97c8p+1f;

  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_FLOATN ay = __clc_fabs(y);
  __CLC_FLOATN v = __clc_fmin(ax, ay);
  __CLC_FLOATN u = __clc_fmax(ax, ay);

  __CLC_FLOATN vbyu;
  if (__clc_denormals_are_zero_fp32()) {
    __CLC_FLOATN s = u > 0x1.0p+96f ? 0x1.0p-32f : 1.0f;
    vbyu = s * __clc_div_fast(v, s * u);
  } else {
    vbyu = v / u;
  }

  __CLC_FLOATN a = __clc_atan_reduced(vbyu);

  __CLC_FLOATN t = piby2 - a;
  a = ay > ax ? t : a;
  t = pi - a;
  a = x < 0.0f ? t : a;

  t = __clc_signbit(x) ? pi : 0.0f;
  a = y == 0.0f ? t : a;

  // x and y are +- Inf
  t = x < 0.0f ? threepiby4 : piby4;
  a = (__clc_isinf(x) & __clc_isinf(y)) ? t : a;

  // x or y is NaN
  a = __clc_isunordered(x, y) ? FLT_NAN : a;

  return __clc_copysign(a, y);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_DOUBLEN __clc_atan2(__CLC_DOUBLEN y,
                                                            __CLC_DOUBLEN x) {
  const __CLC_DOUBLEN pi = 0x1.921fb54442d18p+1;
  const __CLC_DOUBLEN piby2 = 0x1.921fb54442d18p+0;
  const __CLC_DOUBLEN piby4 = 0x1.921fb54442d18p-1;
  const __CLC_DOUBLEN threepiby4 = 0x1.2d97c7f3321d2p+1;

  __CLC_DOUBLEN ay = __clc_fabs(y);
  __CLC_DOUBLEN ax = __clc_fabs(x);
  __CLC_DOUBLEN u = __clc_fmax(ax, ay);
  __CLC_DOUBLEN v = __clc_fmin(ax, ay);
  __CLC_DOUBLEN vbyu = v / u;

  __CLC_DOUBLEN a = __clc_atan_reduced(vbyu);
  __CLC_LONGN xneg = __clc_signbit(x);

  __CLC_DOUBLEN t = piby2 - a;
  a = ax < ay ? t : a;
  t = pi - a;
  a = xneg ? t : a;

  t = xneg ? pi : 0.0;
  a = y == 0.0 ? t : a;

  t = xneg ? threepiby4 : piby4;
  t = __clc_copysign(t, y);
  a = (__clc_isinf(x) && __clc_isinf(y)) ? t : a;

  a = __clc_isunordered(x, y) ? DBL_NAN : a;
  return __clc_copysign(a, y);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_HALFN __clc_atan2(__CLC_HALFN y,
                                                          __CLC_HALFN x) {
  const __CLC_HALFN pi = 0x1.921fb6p+1h;
  const __CLC_HALFN piby2 = 0x1.921fb6p+0h;
  const __CLC_HALFN piby4 = 0x1.921fb6p-1h;
  const __CLC_HALFN threepiby4 = 0x1.2d97c8p+1h;

  __CLC_HALFN ax = __clc_fabs(x);
  __CLC_HALFN ay = __clc_fabs(y);
  __CLC_HALFN v = __clc_fmin(ax, ay);
  __CLC_HALFN u = __clc_fmax(ax, ay);

  __CLC_HALFN vbyu = v / u;

  __CLC_HALFN a = __clc_atan_reduced(vbyu);

  __CLC_HALFN t = piby2 - a;
  a = ay > ax ? t : a;
  t = pi - a;
  a = x < 0.0h ? t : a;

  t = __clc_signbit(x) ? pi : 0.0h;
  a = y == 0.0h ? t : a;

  // x and y are +- Inf
  t = x < 0.0h ? threepiby4 : piby4;
  a = (__clc_isinf(x) && __clc_isinf(y)) ? t : a;

  // x or y is NaN
  a = __clc_isunordered(x, y) ? HALF_NAN : a;

  return __clc_copysign(a, y);
}

#endif
