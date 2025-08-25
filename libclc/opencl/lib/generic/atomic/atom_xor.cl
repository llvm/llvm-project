//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/atomic/atom_xor.h>
#include <clc/opencl/atomic/atomic_xor.h>

#ifdef cl_khr_global_int32_extended_atomics
#define __CLC_ATOMIC_OP xor
#define __CLC_ATOMIC_ADDRESS_SPACE global
#include "atom_int32_binary.inc"
#endif // cl_khr_global_int32_extended_atomics

#ifdef cl_khr_local_int32_extended_atomics
#define __CLC_ATOMIC_OP xor
#define __CLC_ATOMIC_ADDRESS_SPACE local
#include "atom_int32_binary.inc"
#endif // cl_khr_local_int32_extended_atomics

#ifdef cl_khr_int64_extended_atomics

#define __CLC_IMPL(AS, TYPE)                                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_xor(volatile AS TYPE *p, TYPE val) {        \
    return __sync_fetch_and_xor_8(p, val);                                     \
  }

__CLC_IMPL(global, long)
__CLC_IMPL(global, unsigned long)
__CLC_IMPL(local, long)
__CLC_IMPL(local, unsigned long)
#undef __CLC_IMPL

#endif // cl_khr_int64_extended_atomics
