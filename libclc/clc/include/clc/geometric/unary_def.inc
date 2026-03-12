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

// Geometric functions are only defined for scalar, vec2, vec3 and vec4
#if (__CLC_VECSIZE_OR_1 == 1 || __CLC_VECSIZE_OR_1 == 2 ||                     \
     __CLC_VECSIZE_OR_1 == 3 || __CLC_VECSIZE_OR_1 == 4)

_CLC_OVERLOAD _CLC_DEF
#ifdef __CLC_GEOMETRIC_RET_GENTYPE
    __CLC_GENTYPE
#else
    __CLC_SCALAR_GENTYPE
#endif
    __CLC_FUNCTION(__CLC_GENTYPE a) {
  return __CLC_IMPL_FUNCTION(__CLC_FUNCTION)(a);
}

#endif
