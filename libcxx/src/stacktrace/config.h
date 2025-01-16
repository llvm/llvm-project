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

// Check for unwind.h -- could exist on any OS (in theory), but it (or `libunwind`) is likely on Linux systems, and also
// comes with XCode tools on MacOS.
#if __has_include(<unwind.h>)
#  define _LIBCPP_STACKTRACE_UNWIND_IMPL
#endif

// Whether we can invoke external processes via `posix_spawn`
#if __has_include(<spawn.h>) && _LIBCPP_STACKTRACE_ALLOW_TOOLS_AT_RUNTIME
#  define _LIBCPP_STACKTRACE_CAN_SPAWN_TOOLS
#endif

#endif // _LIBCPP_STACKTRACE_CONFIG_H
