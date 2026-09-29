//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_bitfield_insert(__CLC_GENTYPE base,
                                                           __CLC_GENTYPE insert,
                                                           uint offset,
                                                           uint count) {
  __CLC_U_GENTYPE u_base = __CLC_AS_U_GENTYPE(base);
  __CLC_U_GENTYPE u_insert = __CLC_AS_U_GENTYPE(insert);
  __CLC_U_GENTYPE mask = (((__CLC_U_GENTYPE)1 << count) - (__CLC_U_GENTYPE)1)
                         << offset;
  mask = count < __CLC_GENSIZE ? mask : ~(__CLC_U_GENTYPE)0;
  __CLC_U_GENTYPE result = ((u_insert << offset) & mask) | (u_base & ~mask);
  return __CLC_AS_GENTYPE(result);
}
