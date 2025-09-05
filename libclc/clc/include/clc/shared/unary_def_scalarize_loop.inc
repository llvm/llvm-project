//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 >= 2

#include <clc/utils.h>

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION __CLC_FUNCTION
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x) {
  __CLC_GENTYPE result;
  __CLC_SCALAR_GENTYPE *a = (__CLC_SCALAR_GENTYPE *)&x;
  __CLC_SCALAR_GENTYPE *b = (__CLC_SCALAR_GENTYPE *)&result;
  for (int i = 0; i < __CLC_VECSIZE_OR_1; ++i)
    b[i] = __CLC_IMPL_FUNCTION(a[i]);
  return result;
}

#endif // __CLC_VECSIZE_OR_1 >= 2
