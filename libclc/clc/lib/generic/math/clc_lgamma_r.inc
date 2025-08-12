//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_lgamma_r(__CLC_GENTYPE x, __CLC_ADDRSPACE __CLC_INTN *iptr) {
  __CLC_INTN private_iptr;
  __CLC_GENTYPE ret = __clc_lgamma_r(x, &private_iptr);
  *iptr = private_iptr;
  return ret;
}
