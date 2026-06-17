//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_SYNC_LITE_H
#define _LIBCPP___ATOMIC_ATOMIC_SYNC_LITE_H

#include <__atomic/contention_t.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#  if !_LIBCPP_AVAILABILITY_HAS_NEW_SYNC
// Old dylib interface kept for backwards compatibility.
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_wait(void const volatile*, __cxx_contention_t) _NOEXCEPT;
#  endif // !_LIBCPP_AVAILABILITY_HAS_NEW_SYNC

// New dylib interface.
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__atomic_monitor_global(void const* __address) _NOEXCEPT;

_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_global_table(void const* __address, __cxx_contention_t __monitor_value) _NOEXCEPT;

_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_one_global_table(void const*) _NOEXCEPT;
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_all_global_table(void const*) _NOEXCEPT;

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

#endif // _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_SYNC_LITE_H
