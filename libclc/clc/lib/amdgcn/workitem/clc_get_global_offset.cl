//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/workitem/clc_get_global_offset.h>

_CLC_DEF _CLC_OVERLOAD size_t __clc_get_global_offset(uint dim) {
  __constant uint *ptr = (__constant uint *)__builtin_amdgcn_implicitarg_ptr();
  if (dim < 3)
    return ptr[dim + 1];
  return 0;
}
