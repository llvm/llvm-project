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

#if !defined(_LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR)
#  if _LIBCPP_HAS_THREADS &&                                                                                           \
      (((defined(__x86_64__) || defined(_M_X64)) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)) ||                   \
       ((defined(__aarch64__) || defined(_M_ARM64)) && defined(__ARM_FEATURE_ATOMICS)))
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
