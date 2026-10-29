//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_acosh(__CLC_FLOATN x) {
  __CLC_INTN b = x >= 0x1.0p+64f;
  __CLC_FLOATN s = b ? 0x1.0p-64f : 1.0f;
  __CLC_FLOATN sx = x * s;
  __CLC_EP_PAIR a =
      __clc_ep_add(sx, __clc_ep_sqrt(__clc_ep_sub(__clc_ep_sqr(sx), s * s)));
  __CLC_FLOATN z = __clc_ep_ln_hi(a, b ? 64 : 0);

  z = x == __CLC_GENTYPE_INF ? x : z;
  return x < 1.0f ? FLT_NAN : z;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_acosh(__CLC_DOUBLEN x) {
  __CLC_LONGN b = x >= 0x1.0p+512;
  __CLC_DOUBLEN s = b ? 0x1.0p-512 : 1.0;
  __CLC_DOUBLEN sx = x * s;
  __CLC_EP_PAIR a =
      __clc_ep_add(sx, __clc_ep_sqrt(__clc_ep_sub(__clc_ep_sqr(sx), s * s)));
  __CLC_DOUBLEN z = __clc_ep_ln_hi(a, __CLC_CONVERT_INTN(b) ? 512 : 0);

  z = x == __CLC_GENTYPE_INF ? x : z;
  return x < 1.0 ? DBL_NAN : z;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_acosh(__CLC_HALFN hx) {
  __CLC_FLOATN x = __CLC_CONVERT_FLOATN(hx);
  __CLC_FLOATN t = x + __clc_sqrt_fast(__clc_mad(x, x, -1.0f));
  __CLC_HALFN ret = __CLC_CONVERT_HALFN(__clc_log2_fast(t) * 0x1.62e430p-1f);

  return hx < 1.0h ? HALF_NAN : ret;
}

#endif
