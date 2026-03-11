//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/atomic/clc_atomic_fetch_add.h"
#include "clc/opencl/utils.h"

#define __CLC_FUNCTION atomic_fetch_add
#define __CLC_IMPL_FUNCTION __clc_atomic_fetch_add

#define __CLC_BODY "atomic_def.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "atomic_def.inc"
#include "clc/math/gentype.inc"

#if defined(__opencl_c_atomic_order_seq_cst) &&                                \
    defined(__opencl_c_atomic_scope_device)

_CLC_OVERLOAD _CLC_DEF uintptr_t
atomic_fetch_add(volatile __local atomic_uintptr_t *p, ptrdiff_t v) {
  return __scoped_atomic_fetch_add((volatile __local uintptr_t *)p, v,
                                   __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

_CLC_OVERLOAD _CLC_DEF uintptr_t
atomic_fetch_add(volatile __global atomic_uintptr_t *p, ptrdiff_t v) {
  return __scoped_atomic_fetch_add((volatile __global uintptr_t *)p, v,
                                   __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_OVERLOAD _CLC_DEF uintptr_t atomic_fetch_add(volatile atomic_uintptr_t *p,
                                                  ptrdiff_t v) {
  return __scoped_atomic_fetch_add((volatile uintptr_t *)p, v, __ATOMIC_SEQ_CST,
                                   __MEMORY_SCOPE_DEVICE);
}

#endif // _CLC_GENERIC_AS_SUPPORTED

#endif // defined(__opencl_c_atomic_order_seq_cst) &&
       // defined(__opencl_c_atomic_scope_device)
