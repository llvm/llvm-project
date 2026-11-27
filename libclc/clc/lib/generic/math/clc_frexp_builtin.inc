//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CLC_PRIVATE) && defined(__CLC_SCALAR)

#if __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_frexp(__CLC_GENTYPE x,
                                                 private __CLC_INTN *ep) {
  return __builtin_frexpf16(x, ep);
}

#elif __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_frexp(__CLC_GENTYPE x,
                                                 private __CLC_INTN *ep) {
  return __builtin_frexpf(x, ep);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_frexp(__CLC_GENTYPE x,
                                                 private __CLC_INTN *ep) {
  return __builtin_frexp(x, ep);
}

#else
#error "unhandled fpsize"
#endif

#elif !defined(__CLC_PRIVATE)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_INTN e;
  __CLC_GENTYPE result = __clc_frexp(x, &e);
  *ep = e;
  return result;
}

#endif
