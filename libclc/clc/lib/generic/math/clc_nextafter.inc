//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_GENTYPE
__clc_nextafter(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  __CLC_GENTYPE up = __clc_nextup(x);
  __CLC_GENTYPE down = __clc_nextdown(x);
  __CLC_GENTYPE ret = x < y ? up : y;
  ret = x > y ? down : ret;

  return __clc_isunordered(x, y) ? __CLC_GENTYPE_NAN : ret;
}
