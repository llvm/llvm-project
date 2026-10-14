//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN __clc_atanpi(__CLC_FLOATN x) {
  __CLC_FLOATN v = __clc_fabs(x);
  __CLC_INTN g = v > 1.0f;

  __CLC_FLOATN vi = __clc_recip_fast(v);
  v = g ? vi : v;

  __CLC_FLOATN a = __clc_atanpi_reduced(v);

  __CLC_FLOATN y = 0.5f - a;
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_DOUBLEN __clc_atanpi(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN v = __clc_fabs(x);
  __CLC_LONGN g = v > 1.0;

  v = g ? (1.0 / v) : v;

  __CLC_DOUBLEN a = __clc_atanpi_reduced(v);
  __CLC_DOUBLEN y = 0.5 - a;
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_HALFN __clc_atanpi(__CLC_HALFN x) {
  __CLC_HALFN v = __clc_fabs(x);
  __CLC_SHORTN g = v > 1.0h;

  __CLC_HALFN vi = __clc_recip_fast(v);
  v = g ? vi : v;

  __CLC_HALFN a = __clc_atanpi_reduced(v);

  __CLC_HALFN y = 0.5h - a;
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#endif
