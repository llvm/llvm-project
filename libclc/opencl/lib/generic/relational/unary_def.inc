//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#define __CLC_IMPL_FUNCTION(x) __CLC_CONCAT(__clc_, x)

#if __CLC_VECSIZE_OR_1 == 1
#define __CLC_RETTYPE __CLC_INTN
#else
#define __CLC_RETTYPE __CLC_BIT_INTN
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_RETTYPE __CLC_FUNCTION(__CLC_GENTYPE a) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(a);
}

#undef __CLC_RETTYPE
