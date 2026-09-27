//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_DOUBLEN __clc_get_twobypi_bits(__CLC_DOUBLEN x,
                                                            __CLC_INTN index) {
  const __CLC_INTN e_clamp = 1077;
  __CLC_INTN e = __CLC_CONVERT_INTN(__CLC_AS_ULONGN(x) >> 52);
  __CLC_INTN shift = e > e_clamp ? e - e_clamp : 0;
  __CLC_INTN scale = e >= 0x7b0 ? 128 : 0;

  __CLC_INTN start = shift + index * 53;

  __CLC_INTN i = start >> 5;
  __CLC_INTN b = start & 0x1f;

  __CLC_UINTN w2 = __CLC_USE_TABLE(two_by_pi_bits_tbl, i);
  __CLC_UINTN w1 = __CLC_USE_TABLE(two_by_pi_bits_tbl, i + 1);
  __CLC_UINTN w0 = __CLC_USE_TABLE(two_by_pi_bits_tbl, i + 2);

  __CLC_UINTN t = (w2 << b) | (w1 >> (32 - b));
  w2 = b != 0 ? t : w2;

  t = (w1 << b) | (w0 >> (32 - b));
  w1 = b != 0 ? t : w1;
  w1 &= 0xfffff800;

  __CLC_INTN z = __CLC_CONVERT_INTN(__clc_clz(w2));
  b = 11 - z;
  w1 = (w1 >> b) | (w2 << (32 - b));
  w2 >>= b;

  return __CLC_AS_DOUBLEN(
      (__CLC_CONVERT_ULONGN(1022 + scale - start - z) << 52) |
      (__CLC_CONVERT_ULONGN(w2 & 0x000fffff) << 32) | __CLC_CONVERT_ULONGN(w1));
}
