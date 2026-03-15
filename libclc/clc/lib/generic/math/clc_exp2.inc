//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp2(__CLC_GENTYPE x) {
  // Reduce x
  const __CLC_GENTYPE ln2HI = 0x1.62e300p-1f;
  const __CLC_GENTYPE ln2LO = 0x1.2fefa2p-17f;

  __CLC_GENTYPE t = __clc_rint(x);
  __CLC_INTN p = __CLC_CONVERT_INTN(t);
  __CLC_GENTYPE tt = x - t;
  __CLC_GENTYPE hi = tt * ln2HI;
  __CLC_GENTYPE lo = tt * ln2LO;

  // Evaluate poly
  t = hi + lo;
  tt = t * t;
  __CLC_GENTYPE v = __clc_mad(
      tt,
      -__clc_mad(
          tt,
          __clc_mad(tt,
                    __clc_mad(tt,
                              __clc_mad(tt, 0x1.637698p-25f, -0x1.bbd41cp-20f),
                              0x1.1566aap-14f),
                    -0x1.6c16c2p-9f),
          0x1.555556p-3f),
      t);

  __CLC_GENTYPE y = 1.0f - (((-lo) - MATH_DIVIDE(t * v, 2.0f - v)) - hi);

  // Scale by 2^p
  __CLC_GENTYPE r = __CLC_AS_FLOATN(__CLC_AS_INTN(y) + (p << 23));

  const __CLC_GENTYPE ulim = 128.0f;
  const __CLC_GENTYPE llim = -126.0f;

  r = x < llim ? 0.0f : r;
  r = x < ulim ? r : __CLC_AS_FLOATN((__CLC_UINTN)0x7f800000);
  return __clc_isnan(x) ? x : r;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp2(__CLC_GENTYPE x) {
  const __CLC_GENTYPE R_LN2 = 0x1.62e42fefa39efp-1; // ln(2)
  const __CLC_GENTYPE R_1_BY_64 = 1.0 / 64.0;

  __CLC_INTN n = __CLC_CONVERT_INTN(x * 64.0);
  __CLC_GENTYPE r = R_LN2 * __clc_fma(-R_1_BY_64, __CLC_CONVERT_GENTYPE(n), x);

  return __clc_exp_helper(x, -1074.0, 1024.0, r, n);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp2(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_exp2(__CLC_CONVERT_FLOATN(x)));
}

#endif
