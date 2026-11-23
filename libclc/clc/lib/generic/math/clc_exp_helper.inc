//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_exp_helper(__CLC_GENTYPE x,
                                                      __CLC_GENTYPE x_min,
                                                      __CLC_GENTYPE x_max,
                                                      __CLC_GENTYPE r,
                                                      __CLC_INTN n) {

  __CLC_INTN j = n & 0x3f;
  __CLC_INTN m = n >> 6;

  // 6 term tail of Taylor expansion of e^r
  __CLC_GENTYPE z2 =
      r * __clc_fma(
              r,
              __clc_fma(r,
                        __clc_fma(r,
                                  __clc_fma(r,
                                            __clc_fma(r, 0x1.6c16c16c16c17p-10,
                                                      0x1.1111111111111p-7),
                                            0x1.5555555555555p-5),
                                  0x1.5555555555555p-3),
                        0x1.0000000000000p-1),
              1.0);

  __CLC_GENTYPE tv0 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_head, j);
  __CLC_GENTYPE tv1 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_tail, j);
  z2 = __clc_fma(tv0 + tv1, z2, tv1) + tv0;

  __CLC_INTN small_value =
      (m < -1022) || ((m == -1022) && __CLC_CONVERT_INTN(z2 < 1.0));

  __CLC_INTN n1 = m >> 2;
  __CLC_INTN n2 = m - n1;
  __CLC_GENTYPE z3 =
      z2 * __CLC_AS_GENTYPE((__CLC_CONVERT_LONGN(n1) + 1023) << 52);
  z3 *= __CLC_AS_GENTYPE((__CLC_CONVERT_LONGN(n2) + 1023) << 52);

  z2 = __clc_ldexp(z2, m);
  z2 = __CLC_CONVERT_LONGN(small_value) ? z3 : z2;

  z2 = __clc_isnan(x) ? x : z2;

  z2 = x > x_max ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64) : z2;
  z2 = x < x_min ? 0.0 : z2;

  return z2;
}
