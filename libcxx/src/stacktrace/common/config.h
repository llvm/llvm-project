//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_CONFIG_H
#define _LIBCPP_STACKTRACE_CONFIG_H

#include <__config>
#include <__config_site>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

#if __has_include(<unwind.h>)
#  define _LIBCPP_STACKTRACE_COLLECT_UNWIND
#endif

#if defined(__APPLE__)
#  define _LIBCPP_STACKTRACE_APPLE
#endif

#if defined(__linux__)
#  define _LIBCPP_STACKTRACE_LINUX
#endif

#if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME
#  define _LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS
#endif

#if __has_include(<windows.h>) && __has_include(<dbghelp.h>) && __has_include(<psapi.h>)
#  define _LIBCPP_STACKTRACE_WINDOWS
#endif

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_CONFIG_H
