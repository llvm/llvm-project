//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/atomic/atom_cmpxchg.h>
#include <clc/opencl/atomic/atomic_cmpxchg.h>

#define __CLC_IMPL(AS, TYPE)                                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_cmpxchg(volatile AS TYPE *p, TYPE cmp,      \
                                           TYPE val) {                         \
    return atomic_cmpxchg(p, cmp, val);                                        \
  }

#ifdef cl_khr_global_int32_base_atomics
__CLC_IMPL(global, int)
__CLC_IMPL(global, unsigned int)
#endif // cl_khr_global_int32_base_atomics
#ifdef cl_khr_local_int32_base_atomics
__CLC_IMPL(local, int)
__CLC_IMPL(local, unsigned int)
#endif // cl_khr_local_int32_base_atomics

#undef __CLC_IMPL

#ifdef cl_khr_int64_base_atomics

#define __CLC_IMPL(AS, TYPE)                                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_cmpxchg(volatile AS TYPE *p, TYPE cmp,      \
                                           TYPE val) {                         \
    return __sync_val_compare_and_swap_8(p, cmp, val);                         \
  }

__CLC_IMPL(global, long)
__CLC_IMPL(global, unsigned long)
__CLC_IMPL(local, long)
__CLC_IMPL(local, unsigned long)
#undef __CLC_IMPL

#endif // cl_khr_int64_base_atomics
