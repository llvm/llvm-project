//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/atomic/atom_min.h>
#include <clc/opencl/atomic/atomic_min.h>

#ifdef cl_khr_global_int32_extended_atomics
#define __CLC_ATOMIC_OP min
#define __CLC_ATOMIC_ADDRESS_SPACE global
#include "atom_int32_binary.inc"
#endif // cl_khr_global_int32_extended_atomics

#ifdef cl_khr_local_int32_extended_atomics
#define __CLC_ATOMIC_OP min
#define __CLC_ATOMIC_ADDRESS_SPACE local
#include "atom_int32_binary.inc"
#endif // cl_khr_local_int32_extended_atomics

#ifdef cl_khr_int64_extended_atomics

unsigned long __clc__sync_fetch_and_min_local_8(volatile local long *, long);
unsigned long __clc__sync_fetch_and_min_global_8(volatile global long *, long);
unsigned long __clc__sync_fetch_and_umin_local_8(volatile local unsigned long *,
                                                 unsigned long);
unsigned long
__clc__sync_fetch_and_umin_global_8(volatile global unsigned long *,
                                    unsigned long);

#define __CLC_IMPL(AS, TYPE, OP)                                               \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_min(volatile AS TYPE *p, TYPE val) {        \
    return __clc__sync_fetch_and_##OP##_##AS##_8(p, val);                      \
  }

__CLC_IMPL(global, long, min)
__CLC_IMPL(global, unsigned long, umin)
__CLC_IMPL(local, long, min)
__CLC_IMPL(local, unsigned long, umin)
#undef __CLC_IMPL

#endif // cl_khr_int64_extended_atomics
