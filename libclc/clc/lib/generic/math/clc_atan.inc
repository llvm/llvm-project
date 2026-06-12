//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_atan(__CLC_FLOATN x) {
  __CLC_FLOATN v = __clc_fabs(x);
  __CLC_INTN g = v > 1.0f;

  __CLC_FLOATN vi = __clc_recip_fast(v);
  v = g ? vi : v;

  __CLC_FLOATN a = __clc_atan_reduced(v);

  __CLC_FLOATN y = __clc_mad(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -a);
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_atan(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN v = __clc_fabs(x);
  __CLC_LONGN g = v > 1.0;

  v = g ? (1.0 / v) : v;

  __CLC_DOUBLEN a = __clc_atan_reduced(v);
  __CLC_DOUBLEN y = __clc_fma(0x1.dd9ad336a0500p-1, 0x1.af154eeb562d6p+0, -a);
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_atan(__CLC_HALFN x) {
  __CLC_HALFN v = __clc_fabs(x);
  __CLC_SHORTN g = v > 1.0h;

  __CLC_HALFN vi = __clc_recip_fast(v);
  v = g ? vi : v;

  __CLC_HALFN a = __clc_atan_reduced(v);

  __CLC_HALFN y = __clc_mad(0x1.ea8p-1h, 0x1.a3cp+0h, -a);
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#endif
