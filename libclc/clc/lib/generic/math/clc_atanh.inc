//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_atanh(__CLC_FLOATN x) {
  __CLC_FLOATN y = __clc_fabs(x);
  __CLC_EP_PAIR a =
      __clc_ep_fast_div(__clc_ep_fast_add(1.0f, y), __clc_ep_fast_sub(1.0f, y));
  __CLC_FLOATN z = 0.5f * __clc_ep_ln_hi(a, 0);
  z = y < 0x1.0p-12f ? y : z;

  z = y > 1.0f ? FLT_NAN : z;
  z = y == 1.0f ? __CLC_GENTYPE_INF : z;
  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_atanh(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN y = __clc_fabs(x);
  __CLC_EP_PAIR a =
      __clc_ep_fast_div(__clc_ep_fast_add(1.0, y), __clc_ep_fast_sub(1.0, y));
  __CLC_DOUBLEN z = 0.5 * __clc_ep_ln_hi(a, 0);
  z = y < 0x1.0p-27 ? y : z;

  z = y > 1.0 ? DBL_NAN : z;
  z = y == 1.0 ? __CLC_GENTYPE_INF : z;

  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_atanh(__CLC_HALFN x) {
  __CLC_HALFN absx = __clc_fabs(x);
  __CLC_FLOATN absxf = __CLC_CONVERT_FLOATN(absx);
  __CLC_FLOATN t = (1.0f + absxf) * __clc_recip_fast(1.0f - absxf);
  __CLC_HALFN ret = __CLC_CONVERT_HALFN(__clc_log2_fast(t) * 0x1.62e430p-2f);
  ret = absx < 0x1.0p-7h ? absx : ret;

  ret = absx == 1.0h ? __CLC_GENTYPE_INF : ret;
  ret = (absx > 1.0h) || __clc_isnan(absx) ? HALF_NAN : ret;

  return __clc_copysign(ret, x);
}

#endif
