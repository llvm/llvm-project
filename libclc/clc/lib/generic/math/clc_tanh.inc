//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

static _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_tanh_small(__CLC_FLOATN y) {
  __CLC_FLOATN y2 = y * y;
  __CLC_FLOATN p = __clc_mad(
      y2,
      __clc_mad(y2,
                __clc_mad(y2, __clc_mad(y2, -0x1.758e7ap-8f, 0x1.521192p-6f),
                          -0x1.b8389cp-5f),
                0x1.110704p-3f),
      -0x1.555532p-2f);
  return __clc_mad(y2, y * p, y);
}

static _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_tanh_large(__CLC_FLOATN y) {
  __CLC_FLOATN t = __clc_exp(2.0f * y);
  return __clc_mad(-2.0f, __clc_recip_fast(t + 1.0f), 1.0f);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_tanh(__CLC_FLOATN x) {
  __CLC_FLOATN y = __clc_fabs(x);
  __CLC_FLOATN z = y < 0.625f ? __clc_tanh_small(y) : __clc_tanh_large(y);
  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_tanh(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN y = __clc_fabs(x);
  __CLC_EP_PAIR e = __clc_ep_exp_extended(__clc_ep_make_pair(y, 0.0));
  __CLC_EP_PAIR ei = __clc_ep_recip(e);
  __CLC_EP_PAIR t =
      __clc_ep_fast_div(__clc_ep_fast_sub(e, ei), __clc_ep_fast_add(e, ei));
  __CLC_DOUBLEN z = t.hi;

  z = y > 19.0625 ? 1.0 : z;
  z = y < 0x1.0p-27 ? y : z;

  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_tanh(__CLC_HALFN hx) {
  __CLC_FLOATN x = __CLC_CONVERT_FLOATN(hx) * (__CLC_FLOATN)M_LOG2E;
  __CLC_FLOATN a = __clc_exp2_fast(x);
  __CLC_FLOATN b = __clc_exp2_fast(-x);
  __CLC_HALFN one = __clc_copysign(1.0h, hx);
  __CLC_HALFN ret = __CLC_CONVERT_HALFN((a - b) * __clc_recip_fast(a + b));
  return __clc_fabs(hx) > 4.5h ? one : ret;
}

#endif
