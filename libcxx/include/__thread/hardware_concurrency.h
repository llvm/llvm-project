// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___THREAD_HARDWARE_CONCURRENCY_H
#define _LIBCPP___THREAD_HARDWARE_CONCURRENCY_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#if _LIBCPP_STD_VER >= 11 && _LIBCPP_HAS_THREADS

// __thread_hardware_concurrency() has the same semantics as std::[j]thread::hardware_concurrency(), but allows querying
// this information without depending on std::thread itself. This function is an alias to
// std::thread::hardware_concurrency(), which is a part of the ABI.

#  ifdef _LIBCPP_ABI_MICROSOFT
#    define _LIBCPP_THREAD_HARDWARE_CONCURRENCY_SYMBOL                                                                 \
      "?hardware_concurrency@thread@" _LIBCPP_TOSTRING(_LIBCPP_ABI_NAMESPACE) "@std@@SAIXZ"
#  else
#    define _LIBCPP_THREAD_HARDWARE_CONCURRENCY_SYMBOL                                                                 \
      _LIBCPP_TOSTRING(__USER_LABEL_PREFIX__)                                                                          \
      "_ZNSt" _LIBCPP_TOSTRING(_LIBCPP_ABI_NAMESPACE_LEN)                                                              \
          _LIBCPP_TOSTRING(_LIBCPP_ABI_NAMESPACE) "6thread20hardware_concurrencyEv"
#  endif

[[__nodiscard__]] _LIBCPP_EXPORTED_FROM_ABI unsigned __thread_hardware_concurrency() _NOEXCEPT
    __asm__(_LIBCPP_THREAD_HARDWARE_CONCURRENCY_SYMBOL);

#endif // _LIBCPP_STD_VER >= 11 && _LIBCPP_HAS_THREADS

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___THREAD_HARDWARE_CONCURRENCY_H
