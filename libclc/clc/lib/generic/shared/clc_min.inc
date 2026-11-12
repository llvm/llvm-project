//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_FPSIZE
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_min(__CLC_GENTYPE a,
                                               __CLC_GENTYPE b) {
  return __clc_fmin(a, b);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_min(__CLC_GENTYPE a,
                                               __CLC_SCALAR_GENTYPE b) {
  return __clc_fmin(a, b);
}
#endif // __CLC_SCALAR

#else // __CLC_FPSIZE

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_min(__CLC_GENTYPE a,
                                               __CLC_GENTYPE b) {
  return (b < a ? b : a);
}

#ifndef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_min(__CLC_GENTYPE a,
                                               __CLC_SCALAR_GENTYPE b) {
  return (b < (__CLC_GENTYPE)a ? (__CLC_GENTYPE)b : a);
}
#endif // __CLC_SCALAR

#endif // __CLC_FPSIZE
