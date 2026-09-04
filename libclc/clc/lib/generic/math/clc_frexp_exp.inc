//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_INTN __clc_frexp_exp(__CLC_GENTYPE x) {
  __CLC_INTN e;
  (void)__clc_frexp(x, &e);
  return e;
}
