//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE bitfield_insert(__CLC_GENTYPE base,
                                                     __CLC_GENTYPE insert,
                                                     uint offset, uint count) {
  return __clc_bitfield_insert(base, insert, offset, count);
}
