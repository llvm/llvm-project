//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 == 3

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cross(__CLC_GENTYPE p0,
                                                 __CLC_GENTYPE p1) {
  return (__CLC_GENTYPE)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                         p0.x * p1.y - p0.y * p1.x);
}

#elif __CLC_VECSIZE_OR_1 == 4

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cross(__CLC_GENTYPE p0,
                                                 __CLC_GENTYPE p1) {
  return (__CLC_GENTYPE)(p0.y * p1.z - p0.z * p1.y, p0.z * p1.x - p0.x * p1.z,
                         p0.x * p1.y - p0.y * p1.x, 0.0F);
}

#endif
