//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_minmag(__CLC_GENTYPE x,
                                                  __CLC_GENTYPE y) {
  __CLC_GENTYPE ax = __clc_fabs(x);
  __CLC_GENTYPE ay = __clc_fabs(y);
  __CLC_GENTYPE ret = ax < ay ? x : __clc_fmin(x, y);
  return ay < ax ? y : ret;
}
