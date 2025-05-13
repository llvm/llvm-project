//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if (__CLC_VECSIZE_OR_1 == 1 || __CLC_VECSIZE_OR_1 == 2 ||                     \
     __CLC_VECSIZE_OR_1 == 3 || __CLC_VECSIZE_OR_1 == 4)

_CLC_OVERLOAD _CLC_DEF __CLC_SCALAR_GENTYPE __clc_distance(__CLC_GENTYPE p0,
                                                           __CLC_GENTYPE p1) {
  return __clc_length(p0 - p1);
}

#endif
