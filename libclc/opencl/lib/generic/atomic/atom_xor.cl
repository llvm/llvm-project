//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/atomic/clc_atomic_fetch_xor.h>
#include <clc/opencl/atomic/atom_xor.h>

// Non-volatile overloads are for backward compatibility with OpenCL 1.0.

#define __CLC_IMPL(AS, TYPE)                                                   \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_xor(volatile AS TYPE *p, TYPE val) {        \
    return __clc_atomic_fetch_xor(p, val, __ATOMIC_RELAXED,                    \
                                  __MEMORY_SCOPE_DEVICE);                      \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF TYPE atom_xor(AS TYPE *p, TYPE val) {                 \
    return atom_xor((volatile AS TYPE *)p, val);                               \
  }

#ifdef cl_khr_global_int32_extended_atomics
__CLC_IMPL(global, int)
__CLC_IMPL(global, unsigned int)
#endif // cl_khr_global_int32_extended_atomics

#ifdef cl_khr_local_int32_extended_atomics
__CLC_IMPL(local, int)
__CLC_IMPL(local, unsigned int)
#endif // cl_khr_local_int32_extended_atomics

#ifdef cl_khr_int64_extended_atomics

__CLC_IMPL(global, long)
__CLC_IMPL(global, unsigned long)
__CLC_IMPL(local, long)
__CLC_IMPL(local, unsigned long)

#endif // cl_khr_int64_extended_atomics
