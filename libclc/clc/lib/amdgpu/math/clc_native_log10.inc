//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_native_log10(__CLC_GENTYPE val) {
  return __clc_native_log2(val) * (M_LN2_F / M_LN10_F);
}
