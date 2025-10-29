//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION(x) __CLC_CONCAT(__clc_, x)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_INTN __CLC_FUNCTION(__CLC_GENTYPE a) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(a);
}
