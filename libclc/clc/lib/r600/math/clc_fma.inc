//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_fma(__CLC_GENTYPE a, __CLC_GENTYPE b,
                                               __CLC_GENTYPE c) {
#if __CLC_FPSIZE == 32
  return __clc_sw_fma(a, b, c);
#else
  return __builtin_elementwise_fma(a, b, c);
#endif
}
