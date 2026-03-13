//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_CONTENTION_T_H
#define _LIBCPP___ATOMIC_CONTENTION_T_H

#include <__atomic/support.h>
#include <__config>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// The original definition of `__cxx_contention_t` seemed a bit arbitrary.
// When we enable the _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE ABI,
// use definitions that are based on what the underlying platform supports
// instead.
#if defined(_LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE)

#  ifdef __linux__
using __cxx_contention_t _LIBCPP_NODEBUG = int32_t;
#  elif defined(__APPLE__)
using __cxx_contention_t _LIBCPP_NODEBUG = int64_t;
#  elif defined(__FreeBSD__) && __SIZEOF_LONG__ == 8
using __cxx_contention_t _LIBCPP_NODEBUG = int64_t;
#  elif defined(_AIX) && !defined(__64BIT__)
using __cxx_contention_t _LIBCPP_NODEBUG = int32_t;
#  elif defined(_WIN32)
using __cxx_contention_t _LIBCPP_NODEBUG = int64_t;
#  else
using __cxx_contention_t _LIBCPP_NODEBUG = int64_t;
#  endif // __linux__

#else // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

#  if defined(__linux__) || (defined(_AIX) && !defined(__64BIT__))
using __cxx_contention_t _LIBCPP_NODEBUG = int32_t;
#  else
using __cxx_contention_t _LIBCPP_NODEBUG = int64_t;
#  endif // __linux__ || (_AIX && !__64BIT__)

#endif // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

using __cxx_atomic_contention_t _LIBCPP_NODEBUG = __cxx_atomic_impl<__cxx_contention_t>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_CONTENTION_T_H
