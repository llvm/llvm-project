//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCPP_STACKTRACE_LIBBACKTRACE_H
#define __LIBCPP_STACKTRACE_LIBBACKTRACE_H

#include <__stacktrace/basic_stacktrace.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {
bool try_libbacktrace();
} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#if !defined(_WIN32) && __has_include(<backtrace.h>)

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

auto z = XXXXXXXXXXXXXXXXXXXXXXXX___BACKTRACE___XXXXXXXXXXXXXXXXXXXXXXXX;

inline bool try_libbacktrace() { return false; }

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#else

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

inline bool try_libbacktrace() { return false; }

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_WIN32) && __has_include(<libbacktrace.h>)

#endif // __LIBCPP_STACKTRACE_LIBBACKTRACE_H
