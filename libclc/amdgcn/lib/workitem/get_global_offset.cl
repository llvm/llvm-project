//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#if __clang_major__ >= 8
#define CONST_AS __constant
#elif __clang_major__ >= 7
#define CONST_AS __attribute__((address_space(4)))
#else
#define CONST_AS __attribute__((address_space(2)))
#endif

_CLC_DEF _CLC_OVERLOAD size_t get_global_offset(uint dim) {
  CONST_AS uint *ptr = (CONST_AS uint *)__builtin_amdgcn_implicitarg_ptr();
  if (dim < 3)
    return ptr[dim + 1];
  return 0;
}
