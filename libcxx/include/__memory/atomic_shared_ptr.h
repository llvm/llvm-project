// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H
#define _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// __atomic_always_lock_free works in both GCC-compat and MSVC-compat (clang-cl):
//   x86-64  + cx16 (-march=x86-64-v2 or -mcx16) -  CMPXCHG16B
//   AArch64 + LSE (-march=armv8.1-a or later)   - CASP
//   Windows x64 with clang-cl + cx16            - also CMPXCHG16B
// ISA whitelist: big-endian 64-bit (e.g. PowerPC) can pass __atomic_always_lock_free
// but our unsigned __int128 layout (low-64=ptr, high-64=ctrl) requires little-endian.
// _LIBCPP_HAS_INT128 is conservatively 0 in MSVC mode; __extension__ unsigned
// __int128 compiles correctly in clang-cl and is used directly in the impl.
#if !defined(_LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR)
#  if _LIBCPP_HAS_THREADS && sizeof(void*) == 8 && __atomic_always_lock_free(16, (void*)0) &&                          \
      (defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64))
#    define _LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR 1
#  else
#    define _LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR 0
#  endif
#endif

#if _LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR
#  include <__memory/atomic_shared_ptr_lock_free.h>
#else
#  include <__memory/atomic_shared_ptr_lock_based.h>
#endif

#endif // _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H
