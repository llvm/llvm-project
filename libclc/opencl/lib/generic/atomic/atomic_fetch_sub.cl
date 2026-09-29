//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/atomic/clc_atomic_fetch_sub.h"
#include "clc/opencl/utils.h"

#define __CLC_FUNCTION atomic_fetch_sub
#define __CLC_IMPL_FUNCTION __clc_atomic_fetch_sub

#define __CLC_BODY "atomic_def.inc"
#include "clc/integer/gentype.inc"

#define __CLC_BODY "atomic_def.inc"
#include "clc/math/gentype.inc"

// If the device subress space is 64-bits, the data types atomic_intptr_t,
// atomic_uintptr_t, atomic_size_t and atomic_ptrdiff_t are supported if the
// cl_khr_int64_base_atomics and cl_khr_int64_extended_atomics extensions are
// supported and have been enabled.
#if __SIZEOF_POINTER__ < 8 || (defined(cl_khr_int64_base_atomics) &&           \
                               defined(cl_khr_int64_extended_atomics))

#if defined(__opencl_c_atomic_scope_device) &&                                 \
    defined(__opencl_c_atomic_order_seq_cst)

_CLC_DEF _CLC_OVERLOAD uintptr_t
atomic_fetch_sub(volatile __local atomic_uintptr_t *p, ptrdiff_t v) {
  return __scoped_atomic_fetch_sub((volatile __local uintptr_t *)p, v,
                                   __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

_CLC_DEF _CLC_OVERLOAD uintptr_t
atomic_fetch_sub(volatile __global atomic_uintptr_t *p, ptrdiff_t v) {
  return __scoped_atomic_fetch_sub((volatile __global uintptr_t *)p, v,
                                   __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

#if _CLC_GENERIC_AS_SUPPORTED
_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub(volatile atomic_uintptr_t *p,
                                                  ptrdiff_t v) {
  return __scoped_atomic_fetch_sub((volatile uintptr_t *)p, v, __ATOMIC_SEQ_CST,
                                   __MEMORY_SCOPE_DEVICE);
}
#endif // _CLC_GENERIC_AS_SUPPORTED
#endif // defined(__opencl_c_atomic_scope_device) &&
       // defined(__opencl_c_atomic_order_seq_cst)

#ifdef __opencl_c_atomic_scope_device

_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub_explicit(
    volatile __local atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
  return __scoped_atomic_fetch_sub((volatile __local uintptr_t *)p, v, order,
                                   __MEMORY_SCOPE_DEVICE);
}

_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub_explicit(
    volatile __global atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
  return __scoped_atomic_fetch_sub((volatile __global uintptr_t *)p, v, order,
                                   __MEMORY_SCOPE_DEVICE);
}

#if _CLC_GENERIC_AS_SUPPORTED
_CLC_DEF _CLC_OVERLOAD uintptr_t atomic_fetch_sub_explicit(
    volatile atomic_uintptr_t *p, ptrdiff_t v, memory_order order) {
  return __scoped_atomic_fetch_sub((volatile uintptr_t *)p, v, order,
                                   __MEMORY_SCOPE_DEVICE);
}
#endif // _CLC_GENERIC_AS_SUPPORTED
#endif // __opencl_c_atomic_scope_device

_CLC_DEF _CLC_OVERLOAD uintptr_t
atomic_fetch_sub_explicit(volatile __local atomic_uintptr_t *p, ptrdiff_t v,
                          memory_order order, memory_scope scope) {
  return __scoped_atomic_fetch_sub((volatile __local uintptr_t *)p, v, order,
                                   __opencl_get_clang_memory_scope(scope));
}

_CLC_DEF _CLC_OVERLOAD uintptr_t
atomic_fetch_sub_explicit(volatile __global atomic_uintptr_t *p, ptrdiff_t v,
                          memory_order order, memory_scope scope) {
  return __scoped_atomic_fetch_sub((volatile __global uintptr_t *)p, v, order,
                                   __opencl_get_clang_memory_scope(scope));
}

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_DEF _CLC_OVERLOAD uintptr_t
atomic_fetch_sub_explicit(volatile atomic_uintptr_t *p, ptrdiff_t v,
                          memory_order order, memory_scope scope) {
  return __scoped_atomic_fetch_sub((volatile uintptr_t *)p, v, order,
                                   __opencl_get_clang_memory_scope(scope));
}

#endif // _CLC_GENERIC_AS_SUPPORTED

#endif // __SIZEOF_POINTER__ < 8 || (defined(cl_khr_int64_base_atomics) &&
       // defined(cl_khr_int64_extended_atomics))
