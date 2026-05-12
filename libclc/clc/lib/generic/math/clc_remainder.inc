//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_remainder(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  __CLC_INTN unused_quo;
  return __clc_remquo(x, y, &unused_quo);
}
