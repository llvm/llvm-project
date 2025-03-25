//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

_CLC_DEF _CLC_OVERLOAD uint get_global_offset(uint dim) {
  __attribute__((address_space(7))) uint *ptr =
      (__attribute__((address_space(7)))
       uint *)__builtin_r600_implicitarg_ptr();
  if (dim < 3)
    return ptr[dim + 1];
  return 0;
}
