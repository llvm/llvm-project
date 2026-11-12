//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// hadd = (x+y)>>1
// This can be simplified to x>>1 + y>>1 + (1 if both x and y have the 1s bit
// set) This saves us having to do any checks for overflow in the addition sum
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_hadd(__CLC_GENTYPE x,
                                                __CLC_GENTYPE y) {
  return (x >> (__CLC_GENTYPE)1) + (y >> (__CLC_GENTYPE)1) +
         (x & y & (__CLC_GENTYPE)1);
}
