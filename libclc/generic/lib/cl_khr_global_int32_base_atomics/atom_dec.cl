//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#define IMPL(TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_dec(volatile global TYPE *p) { \
  return atomic_dec(p); \
}

IMPL(int)
IMPL(unsigned int)
