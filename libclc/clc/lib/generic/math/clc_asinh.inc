//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_asinh(__CLC_FLOATN x) {
  __CLC_FLOATN y = __clc_fabs(x);
  __CLC_INTN b = y >= 0x1.0p+64f;
  __CLC_FLOATN s = b ? 0x1.0p-64f : 1.0f;
  __CLC_FLOATN sy = y * s;
  __CLC_EP_PAIR a =
      __clc_ep_add(sy, __clc_ep_sqrt(__clc_ep_add(__clc_ep_sqr(sy), s * s)));
  __CLC_FLOATN z = __clc_ep_ln_hi(a, b ? 64 : 0);

  z = y < 0x1.0p-12f ? y : z;
  z = y == __CLC_GENTYPE_INF ? y : z;

  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_asinh(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN y = __clc_fabs(x);
  __CLC_LONGN b = y >= 0x1.0p+512;
  __CLC_DOUBLEN s = b ? 0x1.0p-512 : 1.0;
  __CLC_DOUBLEN sy = y * s;
  __CLC_EP_PAIR a =
      __clc_ep_add(sy, __clc_ep_sqrt(__clc_ep_add(__clc_ep_sqr(sy), s * s)));
  __CLC_DOUBLEN z = __clc_ep_ln_hi(a, __CLC_CONVERT_INTN(b) ? 512 : 0);

  z = y < 0x1.0p-27 ? y : z;
  z = y == __CLC_GENTYPE_INF ? y : z;

  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_asinh(__CLC_HALFN hx) {
  __CLC_FLOATN x = __CLC_CONVERT_FLOATN(__clc_fabs(hx));
  __CLC_FLOATN t = x + __clc_sqrt_fast(__clc_mad(x, x, 1.0f));
  __CLC_HALFN ret = __clc_copysign(
      __CLC_CONVERT_HALFN(__clc_log2_fast(t) * 0x1.62e430p-1f), hx);

  return __clc_isfinite(hx) ? ret : hx;
}

#endif
