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

#if _LIBCPP_HAS_THREADS

[[__nodiscard__]] _LIBCPP_AVAILABILITY_HARDWARE_CONCURRENCY_FREE_FUNCTION
    _LIBCPP_EXPORTED_FROM_ABI unsigned __thread_hardware_concurrency() _NOEXCEPT;

#endif // _LIBCPP_HAS_THREADS

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___THREAD_HARDWARE_CONCURRENCY_H
