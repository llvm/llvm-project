//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if (__CLC_VECSIZE_OR_1 == 2 || __CLC_VECSIZE_OR_1 == 4 ||                     \
     __CLC_VECSIZE_OR_1 == 8 || __CLC_VECSIZE_OR_1 == 16)

#include <clc/utils.h>

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION(x) __CLC_CONCAT(__clc_, x)
#endif

// The return type is same base type as the input type, with the same vector
// size as the mask. Elements in the mask must be the same size (number of bits)
// as the input value., e.g. char8 ret = shuffle(char2 x, uchar8 mask);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE
__CLC_FUNCTION(__CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 2) x,
               __CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 2) y, __CLC_U_GENTYPE mask) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, y, mask);
}
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE
__CLC_FUNCTION(__CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 4) x,
               __CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 4) y, __CLC_U_GENTYPE mask) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, y, mask);
}
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE
__CLC_FUNCTION(__CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 8) x,
               __CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 8) y, __CLC_U_GENTYPE mask) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, y, mask);
}
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(
    __CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 16) x,
    __CLC_XCONCAT(__CLC_SCALAR_GENTYPE, 16) y, __CLC_U_GENTYPE mask) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(x, y, mask);
}

#endif
