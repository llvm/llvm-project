//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_FLOATN __clc_sinh(__CLC_FLOATN x) {
  const __CLC_EP_PAIR ln2 = __clc_ep_make_pair(__CLC_FP_LIT(0x1.62e430p-1),
                                               __CLC_FP_LIT(-0x1.05c610p-29));
  __CLC_FLOATN absx = __clc_fabs(x);
  __CLC_EP_PAIR e = __clc_ep_exp_extended(__clc_ep_sub(absx, ln2));
  __CLC_EP_PAIR s = __clc_ep_fast_sub(e, __clc_ep_ldexp(__clc_ep_recip(e), -2));
  __CLC_FLOATN z = absx > 0x1.65a9f8p+6f ? __CLC_GENTYPE_INF : s.hi;

  z = absx < 0x1.0p-12f ? absx : z;
  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_DOUBLEN __clc_sinh(__CLC_DOUBLEN x) {
  const __CLC_EP_PAIR ln2 = __clc_ep_make_pair(
      __CLC_FP_LIT(0x1.62e42fefa39efp-1), __CLC_FP_LIT(0x1.abc9e3b39803fp-56));

  __CLC_DOUBLEN absx = __clc_fabs(x);
  __CLC_EP_PAIR e = __clc_ep_exp_extended(__clc_ep_sub(absx, ln2));
  __CLC_EP_PAIR s = __clc_ep_fast_sub(e, __clc_ep_ldexp(__clc_ep_recip(e), -2));
  __CLC_DOUBLEN z = absx >= 0x1.633ce8fb9f87ep+9 ? __CLC_GENTYPE_INF : s.hi;

  z = absx < 0x1.0p-27 ? absx : z;
  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_HALFN __clc_sinh(__CLC_HALFN x) {
  __CLC_FLOATN x_float = __CLC_CONVERT_FLOATN(x) * (__CLC_FLOATN)M_LOG2E;
  __CLC_FLOATN result =
      0.5f * (__clc_exp2_fast(x_float) - __clc_exp2_fast(-x_float));
  return __CLC_CONVERT_HALFN(result);
}

#endif
