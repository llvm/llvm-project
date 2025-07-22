//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#define IMPL(TYPE, AS) \
_CLC_OVERLOAD _CLC_DEF TYPE atomic_sub(volatile AS TYPE *p, TYPE val) { \
  return __sync_fetch_and_sub(p, val); \
}

IMPL(int, global)
IMPL(unsigned int, global)
IMPL(int, local)
IMPL(unsigned int, local)
#undef IMPL
