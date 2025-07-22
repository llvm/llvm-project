//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE
__clc_bitfield_extract_unsigned(__CLC_GENTYPE base, uint offset, uint count) {
  if (count == 0)
    return 0;
  __CLC_U_GENTYPE result = __CLC_AS_U_GENTYPE(base)
                           << (__CLC_GENSIZE - offset - count);
  return result >> (__CLC_GENSIZE - count);
}
