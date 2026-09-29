//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes natural log(1+x). Algorithm based on:
// Ping-Tak Peter Tang
// "Table-driven implementation of the logarithm function in IEEE floating-point
// arithmetic" ACM Transactions on Mathematical Software (TOMS) Volume 16, Issue
// 4 (December 1990)
//
// Note that we use a lookup table of size 64 rather than 128, and compensate by
// having extra terms in the minimax polynomial for the kernel approximation.
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_FLOATN __clc_log1p(__CLC_FLOATN x) {
  __CLC_FLOATN z = __clc_ep_ln_hi(__clc_ep_add(__CLC_FP_LIT(1.0), x), 0);
  z = x == __CLC_GENTYPE_INF ? x : z;
  z = x < -1.0f ? __CLC_GENTYPE_NAN : z;
  z = x == -1.0f ? -__CLC_GENTYPE_INF : z;
  return __clc_fabs(x) < 0x1.0p-24f ? x : z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_DOUBLEN __clc_log1p(__CLC_DOUBLEN x) {
  __CLC_GENTYPE z = __clc_ep_ln_hi(__clc_ep_add(1.0, x), 0);

  z = x == __CLC_GENTYPE_INF ? x : z;
  z = x < -1.0 ? __CLC_GENTYPE_NAN : z;
  z = x == -1.0 ? -__CLC_GENTYPE_INF : z;
  return z;
}


#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_HALFN __clc_log1p(__CLC_HALFN x) {
  __CLC_HALFN ret = __CLC_CONVERT_HALFN(__clc_log2_fast(__CLC_CONVERT_FLOATN(x) + 1.0f) * 0x1.62e430p-1f);
  __CLC_HALFN p = __clc_mad(x, x * __clc_mad(x, 0x1.555556p-2h, -0.5h), x);
  return __clc_fabs(x) < 0x1.0p-6h ? p : ret;
}

#endif
