//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE remquo(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                            private __CLC_INTN *q) {
  return __clc_remquo(x, y, q);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE remquo(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                            global __CLC_INTN *q) {
  return __clc_remquo(x, y, q);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE remquo(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                            local __CLC_INTN *q) {
  return __clc_remquo(x, y, q);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE remquo(__CLC_GENTYPE x, __CLC_GENTYPE y,
                                            generic __CLC_INTN *q) {
  return __clc_remquo(x, y, q);
}
#endif
