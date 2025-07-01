//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp(__CLC_GENTYPE x) {
  // Reduce x
  const __CLC_GENTYPE ln2HI = 0x1.62e300p-1f;
  const __CLC_GENTYPE ln2LO = 0x1.2fefa2p-17f;
  const __CLC_GENTYPE invln2 = 0x1.715476p+0f;

  __CLC_GENTYPE fhalF = x < 0.0f ? -0.5f : 0.5f;
  __CLC_INTN p = __CLC_CONVERT_INTN(__clc_mad(x, invln2, fhalF));
  __CLC_GENTYPE fp = __CLC_CONVERT_GENTYPE(p);
  __CLC_GENTYPE hi = __clc_mad(fp, -ln2HI, x); // t*ln2HI is exact here
  __CLC_GENTYPE lo = -fp * ln2LO;

  // Evaluate poly
  __CLC_GENTYPE t = hi + lo;
  __CLC_GENTYPE tt = t * t;
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
  __CLC_GENTYPE r = __CLC_AS_GENTYPE(__CLC_AS_INTN(y) + (p << 23));

  // ln(largest_normal) = 88.72283905206835305366
  const __CLC_GENTYPE ulim = 0x1.62e430p+6f;
  // ln(smallest_normal) = -87.33654475055310898657
  const __CLC_GENTYPE llim = -0x1.5d589ep+6f;

  r = x < llim ? 0.0f : r;
  r = x < ulim ? r : __CLC_AS_GENTYPE((__CLC_UINTN)0x7f800000);
  return __clc_isnan(x) ? x : r;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp(__CLC_GENTYPE x) {

  const __CLC_GENTYPE X_MIN = -0x1.74910d52d3051p+9;           // -1075*ln(2)
  const __CLC_GENTYPE X_MAX = 0x1.62e42fefa39efp+9;            // 1024*ln(2)
  const __CLC_GENTYPE R_64_BY_LOG2 = 0x1.71547652b82fep+6;     // 64/ln(2)
  const __CLC_GENTYPE R_LOG2_BY_64_LD = 0x1.62e42fefa0000p-7;  // head ln(2)/64
  const __CLC_GENTYPE R_LOG2_BY_64_TL = 0x1.cf79abc9e3b39p-46; // tail ln(2)/64

  __CLC_INTN n = __CLC_CONVERT_INTN(x * R_64_BY_LOG2);
  __CLC_GENTYPE r =
      __clc_fma(-R_LOG2_BY_64_TL, __CLC_CONVERT_GENTYPE(n),
                __clc_fma(-R_LOG2_BY_64_LD, __CLC_CONVERT_GENTYPE(n), x));
  return __clc_exp_helper(x, X_MIN, X_MAX, r, n);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_exp(__CLC_CONVERT_FLOATN(x)));
}

#endif
