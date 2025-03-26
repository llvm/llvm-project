//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE native_powr(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  // x^y == 2^{log2 x^y} == 2^{y * log2 x}
  // for x < 0 propagate nan created by log2
  return native_exp2(y * native_log2(x));
}
