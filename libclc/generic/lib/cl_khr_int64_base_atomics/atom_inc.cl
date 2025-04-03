//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

#ifdef cl_khr_int64_base_atomics

#define IMPL(AS, TYPE) \
_CLC_OVERLOAD _CLC_DEF TYPE atom_inc(volatile AS TYPE *p) { \
  return atom_add(p, (TYPE)1); \
}

IMPL(global, long)
IMPL(global, unsigned long)
IMPL(local, long)
IMPL(local, unsigned long)
#undef IMPL

#endif
