//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/atom_add.h>
#include <clc/atomic/atom_inc.h>
#include <clc/atomic/atomic_inc.h>

#define IMPL(AS, TYPE)                                                         \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_inc(volatile AS TYPE *p) {                  \
    return atomic_inc(p);                                                      \
  }

#ifdef cl_khr_global_int32_base_atomics
IMPL(global, int)
IMPL(global, unsigned int)
#endif // cl_khr_global_int32_base_atomics
#ifdef cl_khr_local_int32_base_atomics
IMPL(local, int)
IMPL(local, unsigned int)
#endif // cl_khr_local_int32_base_atomics

#undef IMPL

#ifdef cl_khr_int64_base_atomics

#define IMPL(AS, TYPE)                                                         \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_inc(volatile AS TYPE *p) {                  \
    return atom_add(p, (TYPE)1);                                               \
  }

IMPL(global, long)
IMPL(global, unsigned long)
IMPL(local, long)
IMPL(local, unsigned long)
#undef IMPL

#endif // cl_khr_int64_base_atomics
