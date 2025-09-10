//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_rsqrt(__CLC_GENTYPE val) {
#pragma clang fp contract(fast)
  return __CLC_FP_LIT(1.0) / __builtin_elementwise_sqrt(val);
}
