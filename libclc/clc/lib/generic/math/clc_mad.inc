//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_mad(__CLC_GENTYPE a, __CLC_GENTYPE b,
                                               __CLC_GENTYPE c) {
#pragma OPENCL FP_CONTRACT ON
  return a * b + c;
}
