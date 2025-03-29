//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#define __CLC_FUNCTION(x) __CLC_CONCAT(__clc_, x)

_CLC_OVERLOAD _CLC_DEF __CLC_INTN FUNCTION(__CLC_FLOATN a, __CLC_FLOATN b) {
  return __CLC_FUNCTION(FUNCTION)(a, b);
}
