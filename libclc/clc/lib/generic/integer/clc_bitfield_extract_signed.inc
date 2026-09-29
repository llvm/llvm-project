//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE
__clc_bitfield_extract_signed(__CLC_GENTYPE base, uint offset, uint count) {
  if (count == 0)
    return 0;
  __CLC_U_GENTYPE x = __CLC_AS_U_GENTYPE(base)
                      << (__CLC_GENSIZE - offset - count);
  // Implement an arithmetic shift right.
  __CLC_U_GENTYPE s = -(x >> (__CLC_GENSIZE - 1));
  __CLC_U_GENTYPE result = ((s ^ x) >> (__CLC_GENSIZE - count)) ^ s;
  return __CLC_AS_S_GENTYPE(result);
}
