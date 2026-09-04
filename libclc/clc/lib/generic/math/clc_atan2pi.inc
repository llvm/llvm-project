//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma OPENCL FP_CONTRACT OFF

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_atan2pi(__CLC_FLOATN y,
                                                             __CLC_FLOATN x) {
  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_FLOATN ay = __clc_fabs(y);
  __CLC_FLOATN v = __clc_fmin(ax, ay);
  __CLC_FLOATN u = __clc_fmax(ax, ay);

  __CLC_FLOATN vbyu = v / u;

  __CLC_FLOATN a = __clc_atanpi_reduced(vbyu);

  __CLC_FLOATN at = 0.5f - a;
  a = ay > ax ? at : a;
  at = 1.0f - a;
  a = x < 0.0f ? at : a;

  at = __clc_signbit(x) ? 1.0f : 0.0f;
  a = y == 0.0f ? at : a;

  // x and y are +- Inf
  at = x < 0.0f ? 0.75f : 0.25f;
  a = (__clc_isinf(x) & __clc_isinf(y)) ? at : a;

  // x or y is NaN
  a = __clc_isunordered(x, y) ? FLT_NAN : a;

  return __clc_copysign(a, y);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_atan2pi(__CLC_DOUBLEN y,
                                                              __CLC_DOUBLEN x) {
  __CLC_DOUBLEN ay = __clc_fabs(y);
  __CLC_DOUBLEN ax = __clc_fabs(x);
  __CLC_DOUBLEN u = __clc_fmax(ax, ay);
  __CLC_DOUBLEN v = __clc_fmin(ax, ay);
  __CLC_DOUBLEN vbyu = v / u;

  __CLC_DOUBLEN a = __clc_atanpi_reduced(vbyu);

  __CLC_LONGN xneg = __clc_signbit(x);

  __CLC_DOUBLEN t = 0.5 - a;
  a = ax < ay ? t : a;
  t = 1.0 - a;
  a = xneg ? t : a;

  t = xneg ? 1.0 : 0.0;
  a = y == 0.0 ? t : a;

  t = xneg ? 0.75 : 0.25;
  t = __clc_copysign(t, y);
  a = (__clc_isinf(x) & __clc_isinf(y)) ? t : a;

  a = __clc_isunordered(x, y) ? DBL_NAN : a;

  return __clc_copysign(a, y);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_atan2pi(__CLC_HALFN y,
                                                            __CLC_HALFN x) {
  __CLC_HALFN ax = __clc_fabs(x);
  __CLC_HALFN ay = __clc_fabs(y);
  __CLC_HALFN v = __clc_fmin(ax, ay);
  __CLC_HALFN u = __clc_fmax(ax, ay);

  __CLC_HALFN vbyu = v / u;

  __CLC_HALFN a = __clc_atanpi_reduced(vbyu);

  __CLC_HALFN at = 0.5h - a;
  a = ay > ax ? at : a;
  at = 1.0h - a;
  a = x < 0.0h ? at : a;

  at = __clc_signbit(x) ? 1.0h : 0.0h;
  a = y == 0.0h ? at : a;

  // x and y are +- Inf
  at = x < 0.0h ? 0.75h : 0.25h;
  a = (__clc_isinf(x) & __clc_isinf(y)) ? at : a;

  // x or y is NaN
  a = __clc_isunordered(x, y) ? HALF_NAN : a;

  return __clc_copysign(a, y);
}

#endif
